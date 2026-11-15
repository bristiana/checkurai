#!/usr/bin/env python3
"""Jury dashboard server: one page that shows what the cell is doing, why, and what it costs.

    python tools/dashboard.py                    # http://127.0.0.1:8770 on the station laptop
    # on the ThinkPad:  ssh -N -L 8770:localhost:8770 intel   then open http://localhost:8770

Reads, never writes:
  /dev/shm/checkurai-live/      live feed from tools/cell_live.py (stages, verdicts, heatmaps, camera, joints)
  results/power-*.csv           tools/power_log.py (read incrementally; the newest file)
  results/bench-*.json          tools/bench_devices.py (the newest complete one)
  config/cell.json              devices and model names when no run is live
Serves  /  (dashboard/index.html),  /api/state  (JSON),  /img/<name>.jpg  (from the live folder).
Standard library only; binds to 127.0.0.1 so it is reachable only through the SSH tunnel.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import statistics as st
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LIVE = Path(os.environ.get("CHECKURAI_LIVE_DIR", "/dev/shm/checkurai-live"))
PAGE = ROOT / "dashboard" / "index.html"
RESULTS = ROOT / "results"
CELL_CFG = ROOT / "config" / "cell.json"
DOMS = ("pkg", "core", "uncore", "dram", "psys")
LIVE_STALE_S = 5.0
VIEW_LABELS = {"direct": "Front", "left": "Left mirror", "right": "Right mirror"}
BOWL = {"place_good": "good bowl", "place_bad": "defect bowl"}
VERDICT_HOLD_S = 6.0  # keep the verdict views on screen this long after the decision, while the arm reaches the spot
# joint HUD scales: 99th percentile of the teleop demos (dataset 63dab1ab…, 31 episodes, 10 Hz):
# speed in degrees per second, strain = |commanded - measured| in degrees (gripper: squeezing a brick shows here)
JOINT_SPEED_P99 = {"shoulder_pan": 79, "shoulder_lift": 125, "elbow_flex": 99, "wrist_flex": 67, "wrist_roll": 48,
                   "gripper": 81}
JOINT_STRAIN_P99 = {"shoulder_pan": 9.0, "shoulder_lift": 16.0, "elbow_flex": 13.9, "wrist_flex": 8.6, "wrist_roll": 6.1,
                    "gripper": 14.6}
JOINT_LABELS = {"shoulder_pan": "Base", "shoulder_lift": "Shoulder", "elbow_flex": "Elbow", "wrist_flex": "Wrist",
                "wrist_roll": "Wrist roll", "gripper": "Gripper"}


# ------------------------------------------------------------------------------------------- power
class PowerTail:
    """Keeps the newest power_log CSV in memory, reading only what was appended since the last call."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.pos = 0
        self.cols: list[str] = []
        self.t: list[float] = []
        self.rows: list[dict] = []
        self.off = dict.fromkeys(DOMS, 0.0)
        self.lock = threading.Lock()

    def _newest(self) -> Path | None:
        hits = sorted(RESULTS.glob("power-*.csv"), key=lambda p: p.stat().st_mtime)
        return hits[-1] if hits else None

    def refresh(self) -> None:
        with self.lock:
            p = self._newest()
            if p is None:
                return
            if p != self.path:
                self.path, self.pos, self.cols, self.t, self.rows = p, 0, [], [], []
                self.off = dict.fromkeys(DOMS, 0.0)
            with p.open("rb") as f:
                f.seek(self.pos)
                chunk = f.read()
            if not chunk:
                return
            end = chunk.rfind(b"\n")
            if end < 0:
                return
            self.pos += end + 1
            for line in chunk[:end].decode(errors="replace").splitlines():
                parts = line.split(",")
                if not parts or not parts[0]:
                    continue
                if parts[0] == "t":
                    self.cols = parts
                    continue
                if not self.cols or len(parts) != len(self.cols):
                    continue
                r: dict[str, Any] = {}
                for k, v in zip(self.cols, parts):
                    if k in ("iso", "gpu_throttle"):
                        r[k] = v
                    else:
                        try:
                            r[k] = float(v) if v != "" else None
                        except ValueError:
                            r[k] = None
                prev = self.rows[-1] if self.rows else None
                for d in DOMS:  # a logger restart resets the counters
                    k = f"{d}_J"
                    if r.get(k) is None:
                        continue
                    if prev is not None and prev.get(k) is not None and r[k] + self.off[d] < prev[k]:
                        self.off[d] = prev[k]
                    r[k] += self.off[d]
                self.rows.append(r)
                self.t.append(r["t"])
            keep = 4 * 3600 * 2  # ~4 h at 0.5 s
            if len(self.rows) > keep * 1.2:
                self.rows, self.t = self.rows[-keep:], self.t[-keep:]

    def latest(self) -> dict | None:
        return self.rows[-1] if self.rows else None

    def window(self, t0: float, t1: float) -> list[dict]:
        i, j = bisect.bisect_left(self.t, t0), bisect.bisect_right(self.t, t1)
        return self.rows[i:j]

    def energy(self, t0: float, t1: float, dom: str) -> tuple[float, float] | None:
        w = self.window(t0, t1)
        if len(w) < 2 or w[0].get(f"{dom}_J") is None:
            return None
        return w[-1][f"{dom}_J"] - w[0][f"{dom}_J"], w[-1]["t"] - w[0]["t"]


power = PowerTail()


def power_block(history: list[dict], idle_w: float | None) -> dict:
    power.refresh()
    last = power.latest()
    if not last or time.time() - last["t"] > 5:
        return {"live": False, "file": power.path.name if power.path else None}
    out: dict[str, Any] = {
        "live": True, "pkg_W": last.get("pkg_W"), "psys_W": last.get("psys_W"), "gpu_W": last.get("uncore_W"),
        "cpu_pct": last.get("cpu_pct"), "pkg_C": last.get("pkg_C"), "hot_C": last.get("hot_C"), "tjmax_C": last.get("tjmax_C"),
        "gpu_mhz": last.get("gpu_mhz"), "gpu_throttle": last.get("gpu_throttle") or None, "npu_pct": last.get("npu_pct"),
        "npu_mhz": last.get("npu_mhz"), "on_mains": last.get("ac") == 1, "idle_W": idle_w,
    }
    if out["tjmax_C"] and out["pkg_C"] is not None:
        out["headroom_C"] = round(out["tjmax_C"] - out["pkg_C"], 1)
    # smooth the "now" numbers over the last 3 s so the big digits don't flicker
    recent = power.window(last["t"] - 3, last["t"])
    for k, src in (("pkg_W", "pkg_W"), ("psys_W", "psys_W"), ("gpu_W", "uncore_W"), ("cpu_pct", "cpu_pct"), ("npu_pct", "npu_pct")):
        vals = [r[src] for r in recent if r.get(src) is not None]
        if vals:
            out[k] = round(st.mean(vals), 1)
    # last 5 minutes at 1 Hz
    w = power.window(last["t"] - 300, last["t"])[::2]
    out["series"] = {"t": [round(r["t"], 1) for r in w], "psys_W": [r.get("psys_W") for r in w],
                     "pkg_W": [r.get("pkg_W") for r in w], "gpu_W": [r.get("uncore_W") for r in w],
                     "pkg_C": [r.get("pkg_C") for r in w], "npu_pct": [r.get("npu_pct") for r in w]}
    # energy per sorted brick (median over this run's sorted cycles); platform (psys) when available
    dom = "psys" if last.get("psys_J") is not None else "pkg"
    per = []
    for h in history:
        if h.get("result") != "sorted" or not h.get("t0"):
            continue
        e = power.energy(h["t0"], h["t1"], dom)
        if e and e[1] > 1:
            per.append((e[0], e[1]))
    if per:
        j = st.median(x[0] for x in per)
        s = st.median(x[1] for x in per)
        out["brick"] = {"n": len(per), "J": round(j, 1), "Wh": round(j / 3600, 4), "s": round(s, 1), "domain": dom,
                        "W": round(j / s, 1) if s else None}
    else:
        out["brick"] = None
    return out


# ------------------------------------------------------------------------------------------- bench
_bench_cache: dict[str, Any] = {"path": None, "mtime": 0, "data": None}

NICE = {"policy": "ACT policy", "anomaly": "Anomaly model"}


def bench_block() -> dict | None:
    hits = sorted(RESULTS.glob("bench-*.json"), key=lambda p: p.stat().st_mtime)
    hits = [p for p in hits if "smoke" not in p.name]
    for p in reversed(hits):
        try:
            if _bench_cache["path"] == p and _bench_cache["mtime"] == p.stat().st_mtime:
                return _bench_cache["data"]
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not d.get("finished"):
            continue
        rows = []
        for m in d.get("models", []):
            label = m["label"]
            name = "ACT policy" if m["role"] == "policy" else ("PatchCore" if "patchcore" in label else
                                                              "EfficientAd" if "efficientad" in label else label)
            for r in m.get("runs", []):
                p_ = r.get("power") or {}
                rows.append({"model": name, "label": label, "role": m["role"], "device": r["device"],
                             "median_ms": r.get("median_ms"), "p95_ms": r.get("p95_ms"), "first_ms": r.get("first_ms"),
                             "compile_s": r.get("compile_s"), "tput_fps": r.get("throughput_fps"),
                             "pkg_W": p_.get("pkg_W_mean"), "gpu_W": p_.get("uncore_W_mean"), "npu_pct": p_.get("npu_pct_mean"),
                             "mJ_above_idle": r.get("pkg_mJ_per_inference_above_idle"), "precision": r.get("precision"),
                             "weights": m.get("weights"), "error": r.get("error"), "contended": bool(r.get("heavy_jobs"))})
        data = {"stamp": d.get("stamp"), "started": d.get("started"), "rows": rows,
                "idle_W": (d.get("idle") or {}).get("pkg_W_mean"), "idle_psys_W": (d.get("idle") or {}).get("psys_W_mean"),
                "system": d.get("system")}
        _bench_cache.update(path=p, mtime=p.stat().st_mtime, data=data)
        return data
    return None


# ------------------------------------------------------------------------------------------- state
def read_live() -> dict | None:
    try:
        return json.loads((LIVE / "live.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def img_url(live: dict, name: str) -> str | None:
    v = (live.get("img") or {}).get(name)
    return f"/img/{name}.jpg?v={v}" if v else None


def cell_cfg() -> dict:
    try:
        return json.loads(CELL_CFG.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def bench_ms(bench: dict | None, role: str, device: str | None) -> float | None:
    if not bench or not device:
        return None
    for r in bench["rows"]:
        if r["role"] == role and r["device"] == device and r.get("median_ms") is not None:
            if role == "anomaly" and r["model"] == "EfficientAd":
                continue
            return r["median_ms"]
    return None


def fmt_score(v: float | None) -> str:
    return "–" if v is None else f"{v:.2f}"


def joints_hud(js: dict, now: float) -> dict | None:
    """Per joint: how fast it moves (from the measured positions) and its strain (how far the measured position lags
    the commanded one), each also as 0..1 of the teleop demos' 99th percentile. Positions only: the SO-101 plugin
    reads no motor current, so this is not power."""
    t, s, a, names = js.get("t") or [], js.get("state") or [], js.get("action") or [], js.get("names") or []
    if len(t) < 3 or now - t[-1] > 2.0 or not names:
        return None
    j0 = max(0, bisect.bisect_left(t, t[-1] - 0.5) - 1)
    dt = max(t[-1] - t[j0], 1e-3)
    out = []
    for k, n in enumerate(names):
        q1, q0 = s[-1][k] if k < len(s[-1]) else None, s[j0][k] if k < len(s[j0]) else None
        speed = abs(q1 - q0) / dt if q1 is not None and q0 is not None else None
        gaps = [abs(a[i][k] - s[i][k]) for i in range(j0, len(t))
                if k < len(a[i]) and a[i][k] is not None and s[i][k] is not None]
        strain = st.mean(gaps) if gaps else None
        sp_n = None if speed is None else min(1.0, speed / JOINT_SPEED_P99.get(n, 90))
        sn_n = None if strain is None else min(1.0, strain / JOINT_STRAIN_P99.get(n, 10))
        out.append({"name": n, "label": JOINT_LABELS.get(n, n), "pos": q1, "speed_dps": None if speed is None else round(speed, 1),
                    "strain_deg": None if strain is None else round(strain, 1),
                    "speed": None if sp_n is None else round(sp_n, 3), "strain": None if sn_n is None else round(sn_n, 3),
                    "moving": bool(speed is not None and speed > 3.0)})
    return {"t": t[-1], "joints": out, "units": {"speed": "°/s", "strain": "° behind the command"}}


# ------------------------------------------------------------------------------------------- scene overlay
scene_watch = None  # tools/scene_overlay.SceneWatcher, started in main()
GOAL = {"pick": "yellow", "place_good": "lilac", "place_bad": "orange"}


def arm_at_home() -> bool:
    """Between stages (no policy running) the arm is at HOME: the only time the bowls' in-zone verdict is measured."""
    lv = read_live() or {}
    return lv.get("stage") not in ("pick", "place_good", "place_bad") and lv.get("motion") in (None, "done")


def scene_block(live: dict | None, alive: bool) -> dict | None:
    if scene_watch is None or not alive:
        return None
    sc = scene_watch.get()
    if not sc:
        return None
    g = (live or {}).get("graph") or {}
    cur = g.get("current")
    dec = (live or {}).get("decision") or {}
    # the bowl the arm is heading to now, and the one the verdict sends the brick to next
    return {**sc, "active": GOAL.get(cur), "next": GOAL.get(dec.get("next")) if cur in ("inspect", "decide") else None}


def focus_block(live: dict | None, alive: bool, now: float) -> dict:
    """What the centre of the screen shows: the top camera, or the inspection while a brick sits on the spot."""
    g = (live or {}).get("graph") or {}
    cur = g.get("current") if alive else None
    vt = g.get("verdict_t")
    if cur == "inspect" or (cur in ("spot_before", "spot_after_pick") and g.get("on_spot")):
        return {"mode": "inspection", "phase": "analysing"}
    if cur == "decide" or (cur in ("place_good", "place_bad") and vt and now - vt < VERDICT_HOLD_S):
        return {"mode": "inspection", "phase": "verdict", "hold_left_s": round(max(0.0, VERDICT_HOLD_S - (now - vt)), 1) if vt else None}
    return {"mode": "top", "phase": None}


def build_state() -> dict:
    now = time.time()
    live = read_live()
    cfg_file = cell_cfg()
    bench = bench_block()
    alive = bool(live and not live.get("ended") and now - live.get("t", 0) < LIVE_STALE_S)
    cfg = {**cfg_file, **((live or {}).get("cfg") or {})}
    act_dev, an_dev = cfg.get("device") or "GPU", cfg.get("anomaly_device") or "GPU"
    history = (live or {}).get("history") or []

    # ---- cell
    stage = live.get("stage") if alive else None
    counts = {"sorted": 0, "good": 0, "defect": 0, "uncertain": 0, "failed": 0}
    for h in history:
        if h.get("result") == "sorted":
            counts["sorted"] += 1
            if h.get("verdict") == "good":
                counts["good"] += 1
            else:
                counts["defect"] += 1
                if h.get("uncertain"):
                    counts["uncertain"] += 1
        else:
            counts["failed"] += 1
    cyc_s = [h["cycle_s"] for h in history if h.get("result") == "sorted" and h.get("cycle_s")]
    status = "idle"
    if live:
        status = "running" if alive else ("error" if live.get("error") else "stopped")
    cyc = (live or {}).get("cycle")
    st_started = live.get("stage_started") if alive else None
    cell = {"status": status, "run_id": (live or {}).get("run_dir"), "cycle": None if cyc is None else cyc + 1, "stage": stage,
            "stage_started": st_started, "stage_elapsed": round(now - st_started, 1) if st_started and stage else None,
            "motion": live.get("motion") if alive else None, "message": (live or {}).get("message"),
            "error": (live or {}).get("error"), "counts": counts,
            "cycle_s_last": cyc_s[-1] if cyc_s else None, "cycle_s_median": round(st.median(cyc_s), 1) if cyc_s else None,
            "live_age_s": round(now - live["t"], 1) if live else None}

    # ---- decision + pipeline
    ins = (live or {}).get("inspection")
    dec = (live or {}).get("decision")
    reason_text = None
    if dec and dec.get("verdict"):
        v, s, lo, hi = dec["verdict"], dec.get("score"), dec.get("low"), dec.get("high")
        if v == "empty":
            reason_text = "EMPTY spot → pick again"
        else:
            if v == "good":
                why = f"GOOD ({fmt_score(s)} < {fmt_score(lo)})"
            elif dec.get("uncertain"):
                why = f"UNCERTAIN ({fmt_score(s)} between {fmt_score(lo)} and {fmt_score(hi)}) → treated as defect"
            else:
                why = f"DEFECT ({fmt_score(s)} ≥ {fmt_score(hi if hi is not None else lo)})"
            reason_text = f"{why} → {BOWL.get(dec.get('next'), dec.get('next'))} → {dec.get('policy') or dec.get('next')}"
    act_stage = stage in ("pick", "place_good", "place_bad")
    order = ["perceive", "detect", "reason", "act", "optimize"]
    active = {"spot_check": "perceive", "inspect": "detect", "reason": "reason"}.get(stage or "", "act" if act_stage else None)

    def step_state(k: str) -> str:
        if not alive:
            return "idle"
        if k == "optimize":
            return "done"
        if active is None:
            return "idle"
        return "active" if k == active else ("done" if order.index(k) < order.index(active) else "idle")

    act_ms = bench_ms(bench, "policy", act_dev)
    prec = next((r.get("precision") for r in (bench or {}).get("rows", []) if r["role"] == "policy" and r["device"] == act_dev), None)
    pipeline = [
        {"key": "perceive", "title": "Perceive", "what": "3 cameras · 30 fps", "device": "CPU", "ms": None,
         "state": step_state("perceive")},
        {"key": "detect", "title": "Detect", "what": "PatchCore · Anomalib", "device": an_dev,
         "ms": (ins or {}).get("infer_ms"), "state": step_state("detect")},
        {"key": "reason", "title": "Reason", "what": "verdict → task", "device": "CPU", "ms": None,
         "state": step_state("reason"), "text": reason_text},
        {"key": "act", "title": "Act", "what": "ACT policy · Physical AI Studio", "device": act_dev, "ms": act_ms,
         "ms_source": "benchmark" if act_ms is not None else None, "state": step_state("act"),
         "text": f"{stage.replace('_', ' ')} · {live.get('stage_model')}" if act_stage and live else None},
        {"key": "optimize", "title": "Optimize", "what": f"OpenVINO{' ' + prec.upper() if prec else ''}",
         "device": act_dev, "ms": None, "state": step_state("optimize")},
    ]

    inspection = None
    if ins:
        views = [{"name": v["name"], "label": VIEW_LABELS.get(v["name"], v["name"]), "score": v.get("score"),
                  "raw": v.get("raw"), "regions": v.get("regions") or [],
                  "img": img_url(live, f"view_{v['name']}"), "heat": img_url(live, f"heat_{v['name']}")}
                 for v in ins.get("views", [])]
        nxt = (dec or {}).get("next")
        inspection = {**{k: ins.get(k) for k in ("cycle", "t", "verdict", "uncertain", "score", "low", "high", "frames",
                                                 "infer_ms", "worst_view", "model")},
                      "device": ins.get("device") or an_dev, "views": views,
                      "cycle": None if ins.get("cycle") is None else ins["cycle"] + 1,
                      "task": BOWL.get(nxt) if nxt else None, "policy": (dec or {}).get("policy")}

    hist_out = [{"cycle": h["cycle"] + 1, "verdict": h.get("verdict"), "uncertain": h.get("uncertain"), "score": h.get("score"),
                 "result": h.get("result"), "cycle_s": h.get("cycle_s"),
                 "thumb": img_url(live, h["thumb"]) if h.get("thumb") else None} for h in history[-14:]][::-1]

    camera = None
    if live and (live.get("img") or {}).get("top"):
        try:
            age = round(now - (LIVE / "top.jpg").stat().st_mtime, 1)
        except OSError:
            age = None
        camera = {"top": img_url(live, "top"), "wrist": img_url(live, "wrist"), "age_s": age}

    joints = None
    js = (live or {}).get("joints") or {}
    if js.get("t"):
        joints = {**js, "range": [-100, 100], "gripper_range": [0, 100]}

    idle_w = (bench or {}).get("idle_psys_W") or (bench or {}).get("idle_W")
    thoughts = ((live or {}).get("thoughts") or [])[-30:]
    graph = (live or {}).get("graph")
    if graph:
        graph = {**graph, "cycle": None if graph.get("cycle") is None else graph["cycle"] + 1, "live": alive}
    fr = (live or {}).get("insp_frame")
    if fr and live:  # the inspection camera's frame of the brick on the spot (this cycle only)
        camera = {**(camera or {"top": None, "wrist": None, "age_s": None}),
                  "inspection": img_url(live, "insp"), "inspection_t": fr.get("t"), "inspection_rects": fr.get("rects")}
    return {"now": now, "demo": False, "cell": cell, "pipeline": pipeline, "inspection": inspection, "thoughts": thoughts,
            "graph": graph, "focus": focus_block(live, alive, now), "explanation": (live or {}).get("explanation"),
            "inspect_progress": (live or {}).get("inspect_progress"), "scene": scene_block(live, alive),
            "joints_hud": joints_hud(js, now) if alive else None,
            "history": hist_out, "camera": camera, "joints": joints, "power": power_block(history, idle_w),
            "bench": bench, "config": {"act_device": act_dev, "anomaly_device": an_dev,
                                       "models": {k: cfg.get(f"{k}_model") for k in ("pick", "place_good", "place_bad")},
                                       "anomaly_model": cfg.get("anomaly_model")}}


def clean(o: Any) -> Any:
    """NaN / inf -> None, so the browser's JSON.parse never fails."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


# ------------------------------------------------------------------------------------------- http
IMG_RE = re.compile(r"^/img/([a-z0-9_]{1,40})\.jpg$")


class Handler(BaseHTTPRequestHandler):
    server_version = "checkurai-dashboard"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/v1":  # the first, denser page (fallback)
                self._send(200, (ROOT / "dashboard" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, json.dumps(clean(build_state()), default=str, allow_nan=False).encode(), "application/json")
            elif (m := IMG_RE.match(path)):
                p = LIVE / f"{m.group(1)}.jpg"
                if p.exists():
                    self._send(200, p.read_bytes(), "image/jpeg")
                else:
                    self._send(404, b"", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001 - never kill the server
            try:
                self._send(500, json.dumps({"error": repr(e)}).encode(), "application/json")
            except OSError:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--print", action="store_true", help="print one /api/state and exit")
    ap.add_argument("--page", default=None, help="serve this HTML file at / (default dashboard/index.html)")
    a = ap.parse_args()
    if a.page:
        global PAGE
        PAGE = Path(a.page).resolve()
    if not a.print:
        global scene_watch
        try:
            sys.path.insert(0, str(ROOT / "tools"))
            from scene_overlay import SceneWatcher

            scene_watch = SceneWatcher(0.5, arm_at_home)
        except Exception as e:  # noqa: BLE001 - the page works without the overlay
            print(f"scene overlay off: {e!r}", flush=True)
    if a.print:
        print(json.dumps(build_state(), indent=1, default=str)[:6000])
        return 0
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"dashboard on http://{a.host}:{a.port}  (page {PAGE}, live {LIVE})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
