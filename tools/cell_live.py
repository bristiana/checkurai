#!/usr/bin/env python3
"""Run the cell with a live feed for the jury dashboard (tools/dashboard.py).

Same arguments as tools/orchestrator.py, e.g.

    python tools/cell_live.py run --cycles 20

It imports the orchestrator and wraps a few of its methods at runtime (no edits to orchestrator.py or
inspection.py). The wrappers publish what the cell is doing to a RAM folder (default /dev/shm/checkurai-live):
  live.json      stage events, motion state, the last inspection (verdict, score, thresholds, per-view scores and
                 the unusual regions found in each view's anomaly map), the robot's "thoughts" (first-person lines
                 generated from these real events, no language model), the sorted-brick history, joint state +
                 commanded action (last 15 s at 10 Hz) and a heartbeat
  top.jpg, wrist.jpg               the latest Studio camera frames (5 per second)
  view_<v>.jpg, heat_<v>.jpg       the model input of each inspection view, plain and with its anomaly heatmap
  hist_<n>.jpg                     a thumbnail per inspected brick
Every hook is wrapped in try/except: if the live feed fails, the hook switches itself off and the cell runs on.
"""
from __future__ import annotations

import atexit
import builtins
import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tools import inspection as insp_mod  # noqa: E402
from tools import orchestrator as orch  # noqa: E402

LIVE = Path(os.environ.get("CHECKURAI_LIVE_DIR", "/dev/shm/checkurai-live"))
FRAME_EVERY_S = 0.2   # camera JPEGs to the dashboard
JOINT_EVERY_S = 0.1   # joint samples kept
JOINT_KEEP_S = 15.0
WRITE_EVERY_S = 0.25  # live.json heartbeat
VIEW_NAMES = {"direct": "The front view", "left": "The left mirror", "right": "The right mirror"}
DEV_NAMES = {"GPU": "iGPU", "NPU": "NPU", "CPU": "CPU"}


class Live:
    """The shared state, written atomically to LIVE/live.json."""

    def __init__(self) -> None:
        LIVE.mkdir(parents=True, exist_ok=True)
        for old in LIVE.glob("*"):
            try:
                old.unlink()
            except OSError:
                pass
        self.lock = threading.Lock()
        self.dead: set[str] = set()
        self.state: dict[str, Any] = {
            "pid": os.getpid(), "started": time.time(), "t": time.time(), "ended": None,
            "run_dir": None, "cfg": {}, "cycle": None, "cycle_started": None,
            "stage": None, "stage_started": None, "stage_model": None, "stage_task": None, "motion": None,
            "message": "starting", "error": None,
            "inspection": None, "decision": None, "history": [], "thoughts": [],
            "img": {}, "joints": {"names": [], "t": [], "state": [], "action": []},
            "graph": None, "explanation": None, "insp_frame": None, "inspect_progress": None,
        }
        self.joints: deque = deque(maxlen=int(JOINT_KEEP_S / JOINT_EVERY_S) + 5)
        self.t_frame = self.t_joint = self.t_write = 0.0
        self.phase = "before"  # which spot check comes next: before (the pick) / after_pick / after_place
        self.last_grab: list | None = None  # the frames of the latest inspection-camera grab
        self.prev_outcome: dict[str, str | None] = {}  # pick / place: how the previous run of that stage ended
        self.look = 0                # inspections of this brick so far (2 = the orchestrator's re-inspection)
        self.reinspect_on = True     # the cell's --reinspect-uncertain switch (set at each cycle start)
        self.write(force=True)

    # ---- decision graph: where this brick is in Cell.cycle (node ids as in the dashboard brief)
    def node(self, current: str, **extra: Any) -> None:
        with self.lock:
            g = self.state.get("graph")
            if g is None:
                return
            cur = g.get("current")
            if cur and cur != current and cur not in g["visited"]:
                g["visited"].append(cur)
            g["current"] = current
            g["t"] = round(time.time(), 2)
            g.update(extra)
        self.write(force=True)

    def graph_set(self, **fields: Any) -> None:
        with self.lock:
            if self.state.get("graph") is not None:
                self.state["graph"].update(fields)
        self.write(force=True)

    def publish_inspection_frame(self, frame, kind: str, rects: dict | None = None) -> None:
        """The inspection camera's view of the spot (downscaled), only while a brick is on it. rects: the model's
        view crops {direct|left|right: [x0, y0, x1, y1]} normalised to the frame."""
        import cv2

        h, w = frame.shape[:2]
        if w > 1280:
            frame = cv2.resize(frame, (1280, round(h * 1280 / w)), interpolation=cv2.INTER_AREA)
        self.put_image("insp", _jpeg(frame, 82))
        with self.lock:
            prev = self.state.get("insp_frame") or {}
            self.state["insp_frame"] = {"t": round(time.time(), 2), "kind": kind, "cycle": self.state.get("cycle"),
                                        "rects": rects or prev.get("rects")}

    # ---- safety: a failing hook disables itself and never raises into the cell
    def guard(self, name: str, fn, *a, **kw) -> None:
        if name in self.dead:
            return
        try:
            fn(*a, **kw)
        except Exception:  # noqa: BLE001
            self.dead.add(name)
            print(f"[live] hook {name} disabled after an error (the cell is unaffected):\n{traceback.format_exc(limit=3)}",
                  file=sys.stderr, flush=True)

    def write(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.t_write < WRITE_EVERY_S:
            return
        self.t_write = now
        with self.lock:
            self.state["t"] = now
            js = self.state["joints"]
            rows = list(self.joints)
            js["t"] = [round(r[0], 2) for r in rows]
            js["state"] = [r[1] for r in rows]
            js["action"] = [r[2] for r in rows]
            data = json.dumps(self.state, default=_jsonable)
        tmp = LIVE / ".live.json.tmp"
        tmp.write_text(data)
        os.replace(tmp, LIVE / "live.json")

    def put_image(self, name: str, data: bytes) -> None:
        tmp = LIVE / f".{name}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, LIVE / f"{name}.jpg")
        with self.lock:
            self.state["img"][name] = self.state["img"].get(name, 0) + 1

    def event(self, message: str, **fields: Any) -> None:
        with self.lock:
            self.state.update(fields)
            self.state["message"] = message
        self.write(force=True)

    def think(self, kind: str, text: str) -> None:
        """One first-person line for the dashboard's thought stream (kind: look/decide/act/warn/done/info)."""
        with self.lock:
            th = self.state["thoughts"]
            if th and th[-1]["text"] == text:
                return
            th.append({"t": round(time.time(), 2), "kind": kind, "text": text})
            del th[:-30]
        self.write(force=True)


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


live: Live | None = None


def _jpeg(img_bgr: np.ndarray, q: int = 85) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else b""


def _dev(d: str | None) -> str:
    return DEV_NAMES.get(str(d), str(d))


def _plain_detail(detail: str) -> str:
    if "Sync read" in detail or "status packet" in detail:
        return "I lost contact with my motors (servo bus error)"
    if "websocket" in detail:
        return "the connection to Studio closed"
    if "never left HOME" in detail:
        return "I never started moving"
    if "away from HOME" in detail:
        return "I stopped away from home"
    return detail


# --------------------------------------------------------------------------------------------- hooks
def install() -> None:
    global live
    live = Live()
    L = live

    # 1. Studio observations: camera frames (5 per second) and joints (10 per second).
    orig_obs = orch.StudioSession._on_observation

    def on_observation(self, data: bytes) -> None:
        orig_obs(self, data)
        L.guard("observation", _observation, self, data)

    def _observation(self, data: bytes) -> None:
        now = time.time()
        if now - L.t_frame >= FRAME_EVERY_S:
            L.t_frame = now
            for key in ("top", "wrist"):
                if key in self.frames:
                    L.put_image(key, self.frames[key][1])
        if now - L.t_joint >= JOINT_EVERY_S:
            L.t_joint = now
            n = int.from_bytes(data[:4], "big")
            header = json.loads(data[4:4 + n])
            st = header.get("state") or {}
            act = header.get("action") or {}
            names = L.state["joints"]["names"] or [orch.joint_key(k) for k in st]
            L.state["joints"]["names"] = names
            st_k = {orch.joint_key(k): v for k, v in st.items()}
            act_k = {orch.joint_key(k): v for k, v in act.items()} if isinstance(act, dict) else {}
            L.joints.append((now, [_r(st_k.get(j)) for j in names], [_r(act_k.get(j)) for j in names]))
        L.write()

    orch.StudioSession._on_observation = on_observation

    # 2. Motion state from the orchestrator's own done detector (transitions only).
    orig_update = orch.DoneDetector.update

    def update(self, t, q):
        res = orig_update(self, t, q)
        L.guard("motion", _motion, self, res, q)
        return res

    def _motion(self, res: str, q) -> None:
        m = res
        if res == "moving" and self.home.near(np.asarray(q, dtype=np.float64)):
            m = "settling"
        if m == L.state.get("motion"):
            return
        L.state["motion"] = m
        started = L.state.get("stage_started") or time.time()
        if m == "moving" and self.t_left is not None:
            L.think("act", f"I'm on my way ({time.time() - started:.1f} s after the start).")
        elif m == "settling":
            L.think("act", "Coming back home, settling…")
        elif m == "stalled":
            L.think("warn", "I've stopped away from home — I seem to be stuck.")
        L.write(force=True)

    orch.DoneDetector.update = update

    # 3. Stages (pick / place_good / place_bad): start and end.
    orig_stage = orch.Cell.run_stage

    async def run_stage(self, name: str):
        L.guard("stage_start", _stage_start, self, name)
        st = await orig_stage(self, name)
        L.guard("stage_end", _stage_end, self, st)
        return st

    def _stage_start(self, name: str) -> None:
        model = self.models.get(name, {}).get("name")
        task = getattr(self.a, f"{name}_task", None) or orch.TASKS.get(name)
        dev = _dev(getattr(self.a, "device", None))
        L.event(f"{name}: running {model}", stage=name, stage_started=time.time(), stage_model=model, stage_task=task,
                motion="start")
        grp = "pick" if name == "pick" else "place"
        prev = L.prev_outcome.get(grp)
        key = "retries" if prev == "done" else "rehomes" if prev in ("timeout", "stalled", "failed") else None
        # retries: the stage finished but the spot check said it missed (Cell.cycle's retry loop);
        # rehomes: it timed out / stalled / lost the bus, so run_stage_tries went HOME and runs it again
        if key:
            with L.lock:
                g = L.state.get("graph")
                if g is not None:
                    g[key] = {**g[key], grp: g[key].get(grp, 0) + 1}
        L.node(name, rehoming=False)
        if name == "pick":
            L.think("act", f"Picking a brick from the plate with my learned pick skill ({model}, on the {dev}).")
        else:
            bowl = "good" if name == "place_good" else "defect"
            L.think("act", f"Carrying it to the {bowl} bowl with my {bowl}-bowl skill ({model}, on the {dev}).")

    def _stage_end(self, st) -> None:
        L.event(f"{st.name}: {st.outcome} in {st.run_s:.1f}s{' ' + st.detail if st.detail else ''}",
                stage=None, motion=None,
                last_stage={"name": st.name, "outcome": st.outcome, "run_s": st.run_s, "load_s": st.load_s,
                            "detail": st.detail, "t": time.time()})
        what = {"pick": "Pick", "place_good": "Placing", "place_bad": "Placing"}.get(st.name, st.name)
        L.prev_outcome["pick" if st.name == "pick" else "place"] = st.outcome
        if st.outcome == "done":
            L.think("done", f"Back home and still — {what.lower()} done in {st.run_s:.1f} s.")
            L.phase = "after_pick" if st.name == "pick" else "after_place"
        elif st.outcome == "timeout":
            L.think("warn", f"{what} took too long ({st.run_s:.0f} s).")
        elif st.outcome == "stopped":
            L.think("warn", "Stopped by the operator.")
        else:
            L.think("warn", f"{what} didn't work: {_plain_detail(st.detail) or st.outcome}.")

    orch.Cell.run_stage = run_stage

    # 4. Spot checks (the inspection camera looks at the spot).
    orig_spot = orch.Cell.spot

    def spot(self):
        def _start():
            L.event("checking the inspection spot", stage="spot_check", stage_started=time.time())
            L.node({"before": "spot_before", "after_pick": "spot_after_pick", "after_place": "spot_after_place"}[L.phase])
        L.guard("spot", _start)
        res = orig_spot(self)
        L.guard("spot_end", _spot_end, res)
        return res

    def _spot_end(res) -> None:
        present = bool(res[0])
        L.event(f"spot: {'brick there' if present else 'empty'}", stage=None)
        ph = L.phase
        L.graph_set(on_spot=present, **({"skipped_pick": True} if ph == "before" and present else {}))
        if present and ph in ("before", "after_pick") and L.last_grab:
            L.publish_inspection_frame(L.last_grab[-1], "spot")
        if ph == "before":
            L.think("look", "There's already a brick on the spot — I'll inspect it right away." if present
                    else "The inspection spot is free.")
        elif ph == "after_pick":
            L.think("look" if present else "warn", "The brick is on the spot. Let me take a close look." if present
                    else "The spot is still empty — my grasp must have missed.")
        elif ph == "after_place":
            L.think("done" if not present else "warn", "The spot is clear — the brick is in its bowl." if not present
                    else "The brick is still on the spot — the place missed.")

    orch.Cell.spot = spot

    # 5. Whole cycles: start, result, history.
    orig_cycle = orch.Cell.cycle

    async def cycle(self, i: int):
        def _start():
            L.phase = "before"
            L.prev_outcome = {}
            L.look = 0
            L.reinspect_on = bool(getattr(self.a, "reinspect_uncertain", True))
            with L.lock:
                L.state["graph"] = {"cycle": i, "current": "home_check", "visited": [], "branch": None,
                                    "retries": {"pick": 0, "place": 0}, "rehomes": {"pick": 0, "place": 0},
                                    "rehoming": False, "on_spot": None, "skipped_pick": False, "result": None,
                                    "abort_at": None, "t": round(time.time(), 2)}
                L.state["explanation"] = None
                L.state["insp_frame"] = None
                L.state["inspect_progress"] = None
            L.event(f"cycle {i} started", cycle=i, cycle_started=time.time(), decision=None, run_dir=str(self.run_dir.name))
            L.think("info", f"Brick {i + 1}: first I check whether the inspection spot is free.")
        L.guard("cycle_start", _start)
        rec = await orig_cycle(self, i)
        L.guard("cycle_end", _cycle_end, i, rec)
        return rec

    def _cycle_end(i: int, rec: dict) -> None:
        ins = rec.get("inspection") or {}
        t1 = time.time()
        entry = {"cycle": i, "t0": t1 - (rec.get("cycle_s") or 0), "t1": t1, "result": rec.get("result"),
                 "cycle_s": rec.get("cycle_s"), "verdict": ins.get("verdict"), "uncertain": ins.get("uncertain"),
                 "score": ins.get("score"), "thumb": f"hist_{i}" if f"hist_{i}" in L.state["img"] else None,
                 "stages": [{"name": s["name"], "outcome": s["outcome"], "run_s": s["run_s"]} for s in rec.get("stages", [])]}
        with L.lock:
            L.state["history"] = (L.state["history"] + [entry])[-200:]
        L.event(f"cycle {i}: {rec.get('result')} in {rec.get('cycle_s', 0):.1f}s", stage=None, motion=None)
        res = rec.get("result") or ""
        if res == "sorted":
            L.node("sorted", result="sorted", on_spot=False)
        else:  # the node where it stopped stays current, marked as the abort point
            cur = (L.state.get("graph") or {}).get("current")
            L.graph_set(result="aborted" if res.startswith("aborted") else "failed", abort_at=cur,
                        reason=res.removeprefix("aborted: "))
        if res == "sorted":
            h = L.state["history"]
            good = sum(1 for x in h if x.get("result") == "sorted" and x.get("verdict") == "good")
            bad = sum(1 for x in h if x.get("result") == "sorted" and x.get("verdict") != "good")
            L.think("done", f"Brick {i + 1} sorted in {rec.get('cycle_s', 0):.1f} s. So far: {good} good, {bad} defective.")
        else:
            L.think("warn", f"I stopped brick {i + 1}: {res.removeprefix('aborted: ')}.")

    orch.Cell.cycle = cycle

    # 6. Inspection: capture each view's model input and anomaly map, then publish the median frame's views.
    calls: list[dict] = []
    capture: list[tuple] | None = None

    orig_infer = insp_mod.AnomalyScorer.infer

    def infer(self, x):
        out = orig_infer(self, x)
        if capture is not None:
            capture.append((x, out[1], out[0]))
        return out

    insp_mod.AnomalyScorer.infer = infer

    orig_mv = insp_mod.MultiViewScorer.__call__

    def mv_call(self, img_bgr):
        nonlocal capture
        capture = []
        try:
            res = orig_mv(self, img_bgr)
        finally:
            got, capture = capture, None
        L.guard("mv_capture", _mv_capture, self, res, got)
        L.guard("mv_progress", _mv_progress)
        return res

    def _mv_progress() -> None:  # one call per inspected frame
        with L.lock:
            pr = L.state.get("inspect_progress")
            if pr:
                L.state["inspect_progress"] = {**pr, "done": min(pr["of"], pr["done"] + 1)}
        L.write(force=True)

    def _mv_capture(self, res, got) -> None:
        names = list(self.last.keys())
        if len(got) != len(names):
            return
        calls.append({"combined": float(res[0]), "per": dict(res[1]), "ms": float(res[2]),
                      "views": {n: {"x": g[0], "amap": g[1]} for n, g in zip(names, got)}})
        del calls[:-12]

    insp_mod.MultiViewScorer.__call__ = mv_call

    orig_inspect = orch.AnomalibInspector.inspect

    def inspect(self, frames):
        def _start():
            L.look += 1
            L.event("inspecting the brick" if L.look < 2 else "inspecting the brick again", stage="inspect",
                    stage_started=time.time())
            L.node("inspect", on_spot=True, **({"reinspect": "looking"} if L.look >= 2 else {}))
            with L.lock:
                L.state["inspect_progress"] = {"done": 0, "of": len(frames)}
            if frames:
                L.publish_inspection_frame(frames[len(frames) // 2], "inspect", _view_rects(self))
            views = "3 sides (front + 2 mirrors)" if getattr(self, "views", False) else "the inspection camera"
            nf = f"{len(frames)} frame{'s' if len(frames) != 1 else ''}"
            if L.look >= 2:
                L.think("look", f"Looking again: {nf} more of the same brick, same pose.")
            else:
                what = "defect classifier" if hasattr(self, "_live_orig_inspect") else "anomaly model"
                L.think("look", f"Looking at the brick from {views}, {nf}, with my {what} on the "
                                f"{_dev(getattr(self, 'device', None))}.")
        L.guard("inspect_start", _start)
        calls.clear()
        res = getattr(type(self), "_live_orig_inspect", orig_inspect)(self, frames)
        L.guard("inspect_end", _inspect_end, self, res)
        return res

    def _inspect_end(self, res) -> None:
        look = L.look
        if look >= 2 and res.verdict == "empty":  # the orchestrator then keeps the first look (unsure -> defect)
            L.node("decide", branch="unsure", verdict_t=round(time.time(), 2), reinspect="no brick")
            L.think("warn", "Looking again I didn't see the brick, so the first inspection stands: "
                            "unsure → defect bowl, to be safe.")
            return
        # an uncertain first look: the orchestrator inspects it again before it decides (stay on "inspect")
        pending = bool(res.uncertain and res.verdict != "empty" and look == 1 and L.reinspect_on)
        cyc = L.state.get("cycle")
        info: dict[str, Any] = {"cycle": cyc, "t": time.time(), "verdict": res.verdict, "uncertain": res.uncertain,
                                "score": res.score, "low": res.low, "high": res.high, "frames": res.frames,
                                "infer_ms": res.infer_ms, "device": getattr(self, "device", None), "model": res.model,
                                "views": []}
        per_norm: dict[str, float] = {}
        regions_n = 0
        if res.verdict != "empty" and calls:
            best = min(calls, key=lambda c: abs(c["combined"] - (res.score or 0)))
            norm = getattr(self.scorer, "norm", None) or {}
            for n, v in best["views"].items():
                raw = float(best["per"][n])
                if norm.get(n):  # the decision's scale: (raw - median good) / (max good - median good)
                    med, mx = float(norm[n][0]), float(norm[n][1])
                    per_norm[n] = (raw - med) / max(mx - med, 1e-6)
                else:
                    per_norm[n] = raw
                img, heat, regions = _render_view(v["x"], v["amap"], norm.get(n), res.low or 1.0, res.high or 2.0,
                                                  per_norm[n] if norm.get(n) else None)
                L.put_image(f"view_{n}", _jpeg(img))
                L.put_image(f"heat_{n}", _jpeg(heat))
                regions_n += len(regions)
                info["views"].append({"name": n, "score": round(float(per_norm[n]), 3), "raw": round(raw, 3),
                                      "regions": regions})
            info["worst_view"] = max(per_norm, key=per_norm.get) if per_norm else None
            if cyc is not None and "direct" in best["views"]:
                import cv2
                thumb, _, _ = _render_view(best["views"]["direct"]["x"], None, None, 1.0, 2.0)
                L.put_image(f"hist_{cyc}", _jpeg(cv2.resize(thumb, (160, 160), interpolation=cv2.INTER_AREA), 80))
        place = "place_good" if res.verdict == "good" else "place_bad"
        cell_args = L.state.get("cfg") or {}
        info["look"] = look
        decision = {"verdict": res.verdict, "uncertain": res.uncertain, "score": res.score, "low": res.low, "high": res.high,
                    "next": None if res.verdict == "empty" or pending else place, "look": look,
                    "policy": cell_args.get(f"{place}_model"), "task": cell_args.get(f"{place}_task") or orch.TASKS.get(place)}
        L.event(f"verdict: {res.verdict}{' (uncertain)' if res.uncertain else ''} {res.score}"
                f"{' (look ' + str(look) + ')' if look >= 2 or pending else ''}",
                stage="inspect" if pending else "reason", stage_started=time.time(), inspection=info, decision=decision,
                explanation=_explanation(res, per_norm, regions_n, look, pending))
        if pending:
            L.graph_set(reinspect="pending")
        else:
            branch = "empty" if res.verdict == "empty" else "good" if res.verdict == "good" else \
                "unsure" if res.uncertain else "defect"
            L.node("decide", branch=branch, verdict_t=round(time.time(), 2),
                   **({"reinspect": "done"} if look >= 2 else {}))
        L.think(*_verdict_thought(res, per_norm, regions_n, look, pending))

    orch.AnomalibInspector.inspect = inspect

    # 6b. The supervised defect classifier (tools/brick_classifier.py): the same live hooks; its view crops feed
    # `calls` like MultiViewScorer does (no anomaly map: the views are shown plain, scored by P(defect)).
    try:
        from tools import brick_classifier as bc
    except Exception:  # noqa: BLE001
        bc = None
    if bc is not None:
        cls_orig_frame = bc.BrickClassifierInspector.score_frame

        def cls_score_frame(self, frame):
            s, crops, ms = cls_orig_frame(self, frame)

            def _cap():
                import cv2
                xs = {n: {"x": (cv2.cvtColor(cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_AREA),
                                            cv2.COLOR_BGR2RGB).astype(np.float32) / 255).transpose(2, 0, 1)[None],
                          "amap": None} for n, c in crops.items()}
                calls.append({"combined": float(s), "per": dict(self.last_views), "ms": float(ms), "views": xs})
                del calls[:-12]
            L.guard("cls_capture", _cap)
            return s, crops, ms

        bc.BrickClassifierInspector._live_orig_inspect = bc.BrickClassifierInspector.inspect
        bc.BrickClassifierInspector.inspect = inspect
        bc.BrickClassifierInspector.score_frame = cls_score_frame

    # 7. Remember the configuration once the run starts (models, devices, tasks).
    orig_resolve = orch.Cell.resolve_models

    async def resolve_models(self, stages):
        await orig_resolve(self, stages)
        L.guard("cfg", _cfg, self)

    def _cfg(self) -> None:
        a = self.a
        cfg = {k: getattr(a, k, None) for k in ("device", "anomaly_device", "anomaly_model", "frames", "hz", "execution",
                                                "request_threshold", "backend", "cycles", "only", "reinspect_uncertain",
                                                "reinspect_frames")}
        for st in ("pick", "place_good", "place_bad"):
            cfg[f"{st}_model"] = self.models.get(st, {}).get("name") or getattr(a, f"{st}_model", None)
            cfg[f"{st}_task"] = getattr(a, f"{st}_task", None) or orch.TASKS.get(st)
        L.event("models resolved", cfg=cfg, run_dir=str(self.run_dir.name))
        L.think("info", "Ready: my pick and place skills are loaded, cameras are on.")

    orch.Cell.resolve_models = resolve_models

    # 8. The orchestrator's own status lines (retries, re-homing, operator stop) become thoughts too.
    #    Setting a module-level `print` makes every print() inside orchestrator.py go through this.
    def o_print(*args, **kw):
        builtins.print(*args, **kw)
        if kw.get("file") not in (None, sys.stdout):
            return
        L.guard("print", _from_print, " ".join(str(a) for a in args))

    def _from_print(line: str) -> None:
        if "spot is empty after the pick; retrying" in line:
            L.think("act", "Trying the pick again.")
        elif "still on the spot after the place" in line:
            L.think("act", "Trying to place it again.")
        elif "bringing the arm HOME to try again" in line:
            L.graph_set(rehoming=True)
            L.think("warn", "That didn't go to plan. Going back home to restart this step.")
        elif "could not bring the arm HOME" in line:
            L.graph_set(rehoming=False)
            L.think("warn", "I couldn't get back home — I need a human to help.")
        elif "[cell] arm is HOME" in line:
            L.graph_set(rehoming=False)
            L.think("info", "I'm home again.")
        elif "stop requested" in line:
            L.think("warn", "The operator asked me to stop. Holding still.")
        elif line.startswith("[studio] error"):
            L.think("warn", "Studio reported a problem: " + line.split(":", 1)[-1].strip()[:120])

    orch.print = o_print

    # 9. Inspection camera grabs: keep the latest frames so the spot check can publish what it saw.
    orig_grab = orch.InspectionCamera.grab

    def grab(self, n: int, *a, **kw):
        out = orig_grab(self, n, *a, **kw)
        L.last_grab = out
        return out

    orch.InspectionCamera.grab = grab

    def _ended() -> None:
        try:
            L.event("cell program ended", ended=time.time(), stage=None, motion=None)
        except Exception:  # noqa: BLE001
            pass

    atexit.register(_ended)


def _verdict_thought(res, per: dict[str, float], regions_n: int, look: int = 1, pending: bool = False) -> tuple[str, str]:
    if res.verdict == "empty":
        return "warn", "Hmm, I don't see a brick on the spot."
    lo, hi = res.low or 1.0, res.high or 2.0
    pre = "Looking again: " if look >= 2 else ""
    if not per:
        return "decide", f"{pre}Score {res.score:.2f} → {'look again' if pending else res.verdict.upper()}."
    worst = max(per, key=per.get)
    name, s = VIEW_NAMES.get(worst, worst), per[worst]
    spots = f" ({regions_n} unusual spot{'s' if regions_n != 1 else ''})" if regions_n else ""
    if res.verdict == "good":
        return "decide", (f"{pre}All 3 views look like the good bricks I learned (worst: {s:.2f}, my limit is {lo:.2f}). "
                          "Verdict: GOOD → good bowl.")
    if res.uncertain:
        then = ("I'm not sure yet, so I'll look again." if pending else
                "Still not sure, so to be safe it goes to the defect bowl." if look >= 2 else
                "I'm not sure, so to be safe it goes to the defect bowl.")
        return "decide", (f"{pre}{name} looks a bit unusual{spots}: {s:.2f}, above my good limit {lo:.2f} but below my "
                          f"defect line {hi:.2f}. {then}")
    return "decide", (f"{pre}{name} shows a defect{spots}: {s:.2f}, past my defect line {hi:.2f}. "
                      "Verdict: DEFECT → defect bowl.")


def _view_rects(inspector) -> dict | None:
    """The mirror-booth view crops of the anomaly model, normalised to the frame (for boxes on the dashboard)."""
    sv = getattr(getattr(inspector, "scorer", None), "views", None)
    if not sv or not sv.get("frame"):
        return None
    w, h = sv["frame"][:2]
    return {n: [round(r[0] / w, 4), round(r[1] / h, 4), round(r[2] / w, 4), round(r[3] / h, 4)]
            for n, r in sv["views"].items()}


def _explanation(res, per: dict[str, float], regions_n: int, look: int = 1, pending: bool = False) -> dict[str, Any]:
    """The decision in one line, from the actual numbers (the same rule as AnomalibInspector.inspect and the
    orchestrator's re-inspection: an uncertain first look is looked at again; still uncertain -> defect bowl)."""
    e = _explanation1(res, per, regions_n)
    if pending:
        e.update(text=e["text"].split(" → ")[0] + " → unsure → looking again", bowl=None)
    elif look >= 2 and res.verdict != "empty":
        e["text"] = "Looking again: " + e["text"].replace("→ unsure → defect bowl, to be safe", "→ still unsure → defect bowl")
    e["look"] = look
    return e


def _explanation1(res, per: dict[str, float], regions_n: int) -> dict[str, Any]:
    lo, hi = res.low or 1.0, res.high or 2.0
    labels = {"direct": "Front", "left": "Left mirror", "right": "Right mirror"}
    if res.verdict == "empty":
        return {"text": "No brick on the spot → stop this cycle", "branch": "empty", "bowl": None}
    worst = max(per, key=per.get) if per else None
    s = per[worst] if worst else res.score
    who = labels.get(worst, "Score") if worst else "Score"
    if res.verdict == "good":
        text, branch, bowl = f"{who} {s:.2f}, below the good limit {lo:.2f} → good bowl", "good", "good"
    elif res.uncertain:
        text, branch, bowl = (f"{who} {s:.2f}, above the good limit {lo:.2f}, below the defect line {hi:.2f} "
                              "→ unsure → defect bowl, to be safe"), "unsure", "defect"
    else:
        text, branch, bowl = f"{who} {s:.2f}, past the defect line {hi:.2f} → defect bowl", "defect", "defect"
    return {"text": text, "branch": branch, "bowl": bowl, "view": worst, "view_label": labels.get(worst) if worst else None,
            "score": None if s is None else round(float(s), 3), "low": lo, "high": hi, "regions": regions_n,
            "scores": {k: round(float(v), 3) for k, v in per.items()}}


def _r(v: Any) -> float | None:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return None


def _render_view(x: np.ndarray, amap: np.ndarray | None, norm: list | None, low: float, high: float,
                 score: float | None = None):
    """Model input tensor (1,3,S,S RGB 0..1) -> (BGR image, heatmap overlay, unusual regions).

    Regions are the blobs of the anomaly map above `low` on the decision scale (above the worst good brick seen in
    calibration), as normalised circles {x, y, r, peak}, strongest first (up to 3). The anomaly map is smoothed, so
    its peak sits below the image score the decision uses; with `score` (the view's decision-scale score) the map
    is rescaled so its peak equals that score, which makes the colours and the rings agree with the verdict."""
    import cv2

    rgb = (np.clip(np.asarray(x)[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if amap is None:
        return img, img, []
    a = np.asarray(amap, np.float32).squeeze()
    if a.shape != img.shape[:2]:
        a = cv2.resize(a, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    if norm:  # same scale as the decision: 0 = typical good, 1 = worst good seen in calibration
        med, mx = float(norm[0]), float(norm[1])
        v = (a - med) / max(mx - med, 1e-6)
        vmax = float(v.max())
        if score is not None and score > 0 and vmax > 1e-6:
            v = v * (score / vmax)
    else:
        v = (a - a.min()) / max(float(a.max() - a.min()), 1e-6)
    # colour: transparent below the good level, then up to red at the defect band's top
    t = np.clip((v - 0.4) / max(high - 0.4, 1e-6), 0, 1)
    heat = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (np.clip((v - 0.4) / 0.6, 0, 1) * 0.75)[..., None]
    over = (img.astype(np.float32) * 0.85 * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)
    regions = []
    if norm:
        h, w = v.shape
        n, lab, stats, cent = cv2.connectedComponentsWithStats((v >= low).astype(np.uint8), 8)
        for k in range(1, n):
            bx, by, bw, bh, area = stats[k]
            if area < 4:  # defects can be a few pixels (a marker dot); skip single-pixel noise only
                continue
            peak = float(v[lab == k].max())
            r = min(max(max(bw, bh) / 2 + 0.03 * w, 0.06 * w), 0.35 * w)
            regions.append({"x": round(float(cent[k][0]) / w, 4), "y": round(float(cent[k][1]) / h, 4),
                            "r": round(r / w, 4), "peak": round(peak, 3)})
        regions.sort(key=lambda g: -g["peak"])
        regions = regions[:3]
        if not regions and score is not None and score >= low:  # tiny peak: ring the hottest pixel
            py, px = np.unravel_index(int(np.argmax(v)), v.shape)
            regions = [{"x": round(px / w, 4), "y": round(py / h, 4), "r": 0.06, "peak": round(float(v.max()), 3)}]
    return img, over, regions


def main() -> None:
    install()
    try:
        orch.main()
    except SystemExit as e:
        if live is not None and e.code not in (0, None):
            live.guard("exit", lambda: live.event(f"stopped: {e.code}", error=str(e.code), stage=None, ended=time.time()))
        raise
    except BaseException as e:
        if live is not None:
            live.guard("exit", lambda: live.event(f"crashed: {e!r}"[:300], error=repr(e)[:300], stage=None, ended=time.time()))
        raise


if __name__ == "__main__":
    main()
