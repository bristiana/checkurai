#!/usr/bin/env python3
"""Mock backend for the jury dashboard (stdlib only, dev only).

    python mock_server.py                       # animated demo on :8899
    python mock_server.py --freeze defect_left  # hold one brick result (others: good_real, good_green, defect_direct, uncertain)
    python mock_server.py --status idle         # cell not running: last results + "waiting for the next brick"
    python mock_server.py --no-power            # power log not running
    python mock_server.py --blank               # nothing known yet (every section null)
    python mock_server.py --dump mock/state.json

Serves the same URLs as the real backend: / (index.html), /api/state, /img/<name>.jpg (query ignored).
Every number is mock data and the state says demo: true.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import mimetypes
import os
import random
import statistics
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "mock", "img")
LOW, HIGH = 1.0, 1.86

# ---------------------------------------------------------------- brick scenarios
# per view: (score on the decision scale, raw score, regions [(x, y, r, peak)])
SCEN = {
    "good_real": dict(set=0, verdict="good", unc=False, score=-0.52, worst="left", views={
        "direct": (-1.10, 31.4, []), "left": (-0.52, 36.9, []), "right": (-0.71, 34.8, [])}),
    "defect_left": dict(set=3, verdict="defect", unc=False, score=2.31, worst="left", views={
        "direct": (1.22, 47.5, [(0.40, 0.45, 0.09, 1.22)]),
        "left": (2.31, 58.3, [(0.42, 0.50, 0.11, 2.31), (0.62, 0.30, 0.06, 1.08)]),
        "right": (0.35, 40.2, [])}),
    "good_green": dict(set=1, verdict="good", unc=False, score=0.31, worst="left", views={
        "direct": (0.12, 38.8, []), "left": (0.31, 40.0, []), "right": (-0.24, 36.1, [])}),
    "defect_direct": dict(set=4, verdict="defect", unc=False, score=2.64, worst="direct", views={
        "direct": (2.64, 61.2, [(0.35, 0.365, 0.09, 2.64), (0.57, 0.515, 0.08, 1.95)]),
        "left": (1.30, 48.1, [(0.60, 0.45, 0.08, 1.30)]),
        "right": (0.62, 42.7, [])}),
    "uncertain": dict(set=2, verdict="defect", unc=True, score=1.46, worst="right", views={
        "direct": (0.55, 41.9, []), "left": (0.71, 43.0, []), "right": (1.46, 51.6, [(0.45, 0.55, 0.10, 1.46)])}),
}
ROTATION = ["defect_left", "good_real", "good_green", "defect_direct", "good_real", "uncertain", "good_green"]
VIEW_LABEL = {"direct": "Front", "left": "Left mirror", "right": "Right mirror"}
POLICY = {"pick": "act-pick-v3", "place_good": "act-place-good-v3", "place_bad": "act-place-bad-v3"}

# stage durations (s) of one cycle
STAGES = [("spot_check", 2.4), ("pick", 8.6), ("inspect", 4.4), ("reason", 1.6), ("place", 8.2), ("home", 4.6)]
STEP_OF = {"spot_check": "perceive", "pick": "act", "inspect": "detect", "reason": "reason",
           "place_good": "act", "place_bad": "act", "home": "act"}
STEP_ORDER = ["perceive", "detect", "reason", "act", "optimize"]

# arm poses (deg; gripper = % open)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
POSE = {
    "home": [0, -95, 92, 70, 0, 2],
    "spot": [-28, -38, 40, 82, -6, 38],
    "booth": [24, -60, 58, 38, 4, 8],
    "good": [62, -55, 50, 66, 0, 8],
    "bad": [-62, -52, 48, 64, 0, 8],
}


def smooth(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


class Sim:
    def __init__(self, args):
        self.a = args
        self.t0 = time.time() - 15.0  # pretend the run started a little earlier
        self.base_cycle = 10          # first live cycle number (cycles 1..9 are prefilled history)
        self.rng = random.Random(7)
        self.history = self._prefill()

    # ------------------------------------------------------------ schedule
    def cycle_plan(self, k: int):
        rng = random.Random(1000 + k)
        name = self.a.freeze or ROTATION[k % len(ROTATION)]
        sc = SCEN[name]
        bad = sc["verdict"] != "good"
        stages = []
        for s, d in STAGES:
            d = d * (0.9 + 0.2 * rng.random())
            if s == "place":
                s = "place_bad" if bad else "place_good"
            stages.append((s, d))
        return name, stages

    def where(self, now: float):
        """(cycle k, cycle start, [(stage, start, dur)], index of current stage, scenario) for a time."""
        if not hasattr(self, "_starts") or self._t0 != self.t0:
            self._t0, self._starts, self._plans = self.t0, [self.t0], []
        while not self._plans or self._starts[-1] <= now:  # invariant: len(starts) == len(plans) + 1
            k = len(self._plans)
            name, stages = self.cycle_plan(k)
            self._plans.append((name, stages))
            self._starts.append(self._starts[k] + sum(d for _, d in stages))
        k = max(0, bisect.bisect_right(self._starts, now) - 1)
        k = min(k, len(self._plans) - 1)
        start = self._starts[k]
        name, stages = self._plans[k]
        sched, s0 = [], start
        for s, d in stages:
            sched.append((s, s0, d))
            s0 += d
        cur = next((i for i, (s, st, d) in enumerate(sched) if now < st + d), len(sched) - 1)
        return k, start, sched, cur, name

    # ------------------------------------------------------------ history
    def _prefill(self):
        names = ["good_green", "defect_direct", "good_real", "good_real", "uncertain", "good_green", None, "defect_left", "good_real"]
        out, t = [], self.t0 - 9 * 33.0
        for i, n in enumerate(names):
            cyc = i + 1
            cs = round(31 + 4 * random.Random(cyc).random(), 1)
            if n is None:  # a missed pick: nothing reached the booth
                out.append(dict(cycle=cyc, verdict="empty", uncertain=False, score=None, result="missed pick",
                                cycle_s=cs, thumb=f"/img/set5_thumb.jpg?v=h{cyc}", t=t))
            else:
                sc = SCEN[n]
                out.append(dict(cycle=cyc, verdict=sc["verdict"], uncertain=sc["unc"], score=sc["score"], result="sorted",
                                cycle_s=cs, thumb=f"/img/set{sc['set']}_thumb.jpg?v=h{cyc}", t=t))
            t += cs
        return out

    def live_history(self, k_done: int):
        rows = list(self.history)
        for k in range(k_done):
            name, stages = self.cycle_plan(k)
            sc = SCEN[name]
            rows.append(dict(cycle=self.base_cycle + k, verdict=sc["verdict"], uncertain=sc["unc"], score=sc["score"],
                             result="sorted", cycle_s=round(sum(d for _, d in stages), 1),
                             thumb=f"/img/set{sc['set']}_thumb.jpg?v=c{self.base_cycle + k}"))
        return rows

    # ------------------------------------------------------------ pieces
    def inspection(self, name: str, cycle: int, t: float):
        sc = SCEN[name]
        s = sc["set"]
        views = []
        for v in ("direct", "left", "right"):
            score, raw, regs = sc["views"][v]
            views.append(dict(name=v, label=VIEW_LABEL[v], score=score, raw=raw,
                              img=f"/img/set{s}_view_{v}.jpg?v={cycle}", heat=f"/img/set{s}_heat_{v}.jpg?v={cycle}",
                              regions=[dict(x=x, y=y, r=r, peak=p) for x, y, r, p in regs]))
        bad = sc["verdict"] != "good"
        return dict(cycle=cycle, t=round(t, 2), verdict=sc["verdict"], uncertain=sc["unc"], score=sc["score"],
                    low=LOW, high=HIGH, frames=5, infer_ms=round(790 + 30 * random.Random(cycle).random(), 0), device="GPU",
                    task="defect bowl" if bad else "good bowl", policy=POLICY["place_bad" if bad else "place_good"],
                    worst_view=sc["worst"], views=views)

    def reason_text(self, name: str):
        sc = SCEN[name]
        if sc["verdict"] == "good":
            return f"GOOD ({sc['score']:.2f} < {LOW:.2f}) → good bowl"
        if sc["unc"]:
            return f"UNSURE ({LOW:.2f} ≤ {sc['score']:.2f} < {HIGH:.2f}) → defect bowl"
        return f"DEFECT ({sc['score']:.2f} ≥ {HIGH:.2f}) → defect bowl"

    def thoughts_for(self, k: int, sched, upto: int, name: str):
        sc = SCEN[name]
        cyc = self.base_cycle + k
        vw = VIEW_LABEL[sc["worst"]].lower()
        out = []
        for i, (s, st, d) in enumerate(sched[: upto + 1]):
            if s == "spot_check":
                out.append(("look", st, f"Brick #{cyc}: I can see a brick waiting in the pick-up spot."))
            elif s == "pick":
                out.append(("act", st, f"Picking it up with my learned pick skill ({POLICY['pick']}), planned on the iGPU."))
                if name == "defect_direct" and st + 4.5 < time.time():
                    out.append(("warn", st + 4.5, "The grip looked loose, so I re-planned and closed tighter."))
            elif s == "inspect":
                out.append(("look", st, "Holding it in the mirror booth: 5 frames from 3 sides."))
            elif s == "reason":
                if sc["verdict"] == "good":
                    txt = f"Every side looks like a normal brick (worst {sc['score']:.2f}, under 1.0). It's a good one."
                elif sc["unc"]:
                    txt = f"The {vw} scores {sc['score']:.2f}: between normal and defect. Not sure, so it goes to the defect bowl to be safe."
                else:
                    txt = f"The {vw} shows a spot at {sc['score']:.2f}, past my {HIGH:.2f} defect line. That's a defect."
                out.append(("decide", st, txt))
            elif s.startswith("place"):
                bowl = "defect" if s == "place_bad" else "good"
                out.append(("act", st, f"Placing it in the {bowl} bowl ({POLICY[s]})."))
            elif s == "home":
                total = sum(dd for _, _, dd in sched)
                out.append(("done", st, f"Brick #{cyc} sorted in {total:.0f} s. Back home, waiting for the next brick."))
        return out

    def joints(self, now: float):
        ts, st, ac = [], [], []
        for i in range(150):
            t = now - 15 + i * 0.1
            ts.append(round(t, 2))
            cmd = self.pose_at(t + 0.0)
            meas = self.pose_at(t - 0.35)
            n = random.Random(int(t * 10))
            st.append([round(m + n.uniform(-0.6, 0.6), 1) for m in meas])
            ac.append([round(c, 1) for c in cmd])
        return dict(names=JOINTS, t=ts, state=st, action=ac,
                    range=[[-110, 110], [-100, 100], [-100, 100], [-100, 100], [-160, 160]], gripper_range=[0, 100])

    def pose_at(self, t: float):
        if t < self.t0:
            return list(POSE["home"])
        k, start, sched, cur, name = self.where(t)
        s, st, d = sched[cur]
        u = (t - st) / d
        bad = SCEN[name]["verdict"] != "good"

        def lerp(a, b, w):
            return [x + (y - x) * smooth(w) for x, y in zip(a, b)]

        if s == "spot_check":
            return list(POSE["home"])
        if s == "pick":
            p = lerp(POSE["home"], POSE["spot"], u / 0.55) if u < 0.55 else lerp(POSE["spot"], POSE["booth"], (u - 0.65) / 0.35)
            if u >= 0.55:
                p[5] = 38 - 30 * smooth((u - 0.55) / 0.1)  # close the gripper on the brick
            if 0.55 <= u < 0.65:
                p = list(POSE["spot"])
                p[5] = 38 - 30 * smooth((u - 0.55) / 0.1)
            return p
        if s in ("inspect", "reason"):
            p = list(POSE["booth"])
            p[4] += 6 * math.sin((t - st) * 1.3) if s == "inspect" else 0
            return p
        if s.startswith("place"):
            tgt = POSE["bad" if bad else "good"]
            p = lerp(POSE["booth"], tgt, u / 0.8)
            if u > 0.8:
                p[5] = 8 + 32 * smooth((u - 0.8) / 0.15)
            return p
        # home
        frm = list(POSE["bad" if bad else "good"])
        frm[5] = 40
        return lerp(frm, POSE["home"], u / 0.85)

    def power(self, now: float, sched_stage: str | None):
        busy = {"inspect": 1.0, "reason": 0.4, "pick": 0.55, "place_good": 0.55, "place_bad": 0.55, "home": 0.3, "spot_check": 0.2}
        def at(t):
            if self.a.status != "running" or t < self.t0:
                b = 0.05
            else:
                k, start, sched, cur, name = self.where(t)
                b = busy.get(sched[cur][0], 0.2)
            n = random.Random(int(t * 2)).uniform(-1, 1)
            psys = 36.5 + 13.5 * b + 1.2 * n
            pkg = 19.5 + 10.5 * b + 0.8 * n
            gpu = 2.5 + 9.0 * b + 0.4 * n
            temp = 61 + 7 * b + 0.6 * n
            return psys, pkg, gpu, temp, b
        ser = dict(t=[], psys_W=[], pkg_W=[], pkg_C=[], gpu_W=[], npu_pct=[])
        for i in range(150):
            t = now - 300 + i * 2
            psys, pkg, gpu, temp, _ = at(t)
            ser["t"].append(round(t, 1)); ser["psys_W"].append(round(psys, 1)); ser["pkg_W"].append(round(pkg, 1))
            ser["pkg_C"].append(round(temp, 1)); ser["gpu_W"].append(round(gpu, 2)); ser["npu_pct"].append(0.0)
        psys, pkg, gpu, temp, b = at(now)
        return dict(live=True, pkg_W=round(pkg, 1), psys_W=round(psys, 1), gpu_W=round(gpu, 1),
                    cpu_pct=round(14 + 22 * b + random.Random(int(now)).uniform(-3, 3), 0),
                    pkg_C=round(temp, 0), tjmax_C=100, headroom_C=round(100 - temp, 0),
                    gpu_mhz=1550 if b > 0.3 else 750, gpu_throttle="pl1" if b > 0.3 else "",
                    npu_pct=0.0, npu_mhz=0, on_mains=True, hot_C=round(temp + 4, 0), idle_W=6.4,
                    series=ser)

    # ------------------------------------------------------------ full state
    def state(self):
        now = time.time()
        a = self.a
        if a.blank:
            return dict(now=now, demo=True, cell=None, pipeline=None, inspection=None, history=[], camera=None,
                        joints=None, power=dict(live=False) if a.no_power else None, bench=None, config=None, thoughts=[])
        real_now = now
        if a.status == "idle":
            # the run ended at the close of the cycle that was going when idle was switched on; freeze there
            if getattr(self, "idle_at", None) is None:
                kk, ss, _, _, _ = self.where(now)
                self.idle_at = ss + 0.0 if kk > 0 else now
            now = self.idle_at - 0.01
        else:
            self.idle_at = None
        k, start, sched, cur, name = self.where(now)
        stage, st, dur = sched[cur]
        cyc = self.base_cycle + k
        # the last finished inspection: this cycle's once 'reason' started, else the previous cycle's
        ri = next(i for i, (s, _, _) in enumerate(sched) if s == "reason")
        if cur >= ri:
            insp = self.inspection(name, cyc, sched[ri][1])
        elif k > 0:
            pn, ps = self.cycle_plan(k - 1)
            insp = self.inspection(pn, cyc - 1, start - sum(d for _, d in ps[ri:]))
        else:
            h = self.history[-1]
            insp = self.inspection("good_real", h["cycle"], start - 12)
        hist = self.live_history(k + 1 if a.status == "idle" else k)
        # thoughts: the previous cycle + this one so far
        th = []
        if k > 0:
            pk, pstart, psched, _, pname = self.where(start - 0.01)
            th += self.thoughts_for(pk, psched, len(psched) - 1, pname)
        th += self.thoughts_for(k, sched, cur, name)
        th = [dict(t=round(t, 2), kind=kd, text=tx) for kd, t, tx in sorted(th, key=lambda x: x[1]) if t <= now][-30:]

        # pipeline
        active = STEP_OF.get(stage)
        pipe = []
        sc = SCEN[name]
        for key in STEP_ORDER:
            if key == "optimize":
                state = "done"
            elif key == active:
                state = "active"
            elif stage == "pick":
                state = "done" if key == "perceive" else "idle"
            elif stage in ("place_good", "place_bad", "home"):
                state = "done"
            else:
                state = "done" if STEP_ORDER.index(key) < STEP_ORDER.index(active) else "idle"
            pipe.append(state)
        act_text = None
        if stage in ("pick", "place_good", "place_bad"):
            act_text = f"{stage.replace('_', ' ')} · {POLICY[stage]}"
        elif stage == "home":
            act_text = "going home"
        reason_txt = self.reason_text(name) if cur >= ri else (self.reason_text(self.cycle_plan(k - 1)[0]) if k > 0 else None)
        pipeline = [
            dict(key="perceive", title="Perceive", what="Cameras check the pick-up spot for a brick", device="CPU", ms=14, state=pipe[0], text=None),
            dict(key="detect", title="Detect", what="Anomaly model scores 3 views × 5 frames", device="GPU",
                 ms=insp["infer_ms"], state=pipe[1], text="5 frames × 3 views" if pipe[1] == "active" else None),
            dict(key="reason", title="Reason", what="Score vs. thresholds calibrated on good bricks", device="CPU", ms=0.4,
                 state=pipe[2], text=reason_txt),
            dict(key="act", title="Act", what="Learned ACT policy drives the arm", device="GPU", ms=28.4, ms_source="benchmark",
                 state=pipe[3], text=act_text),
            dict(key="optimize", title="Optimize", what="OpenVINO FP16, compiled for this laptop", device=None, ms=None, state=pipe[4], text=None),
        ]
        motion, elapsed = None, round(now - st, 2)
        if stage in ("pick", "place_good", "place_bad", "home"):
            u = (now - st) / dur
            motion = "start" if u < 0.06 else "moving" if u < 0.85 else "settling" if u < 0.97 else "done"
        sorted_rows = [h for h in hist if h["result"] == "sorted"]
        cs = [h["cycle_s"] for h in sorted_rows]
        cell = dict(status=a.status, run_id="20261115-0412", cycle=cyc, stage=stage, stage_started=round(st, 2),
                    stage_elapsed=elapsed, motion=motion,
                    message={"spot_check": "looking for a brick", "pick": "picking", "inspect": "inspecting",
                             "reason": "deciding", "place_good": "placing: good bowl", "place_bad": "placing: defect bowl",
                             "home": "going home"}[stage],
                    error=None,
                    counts=dict(sorted=len(sorted_rows), good=sum(h["verdict"] == "good" for h in sorted_rows),
                                defect=sum(h["verdict"] == "defect" for h in sorted_rows),
                                uncertain=sum(bool(h["uncertain"]) for h in sorted_rows),
                                failed=sum(h["result"] != "sorted" for h in hist)),
                    cycle_s_last=cs[-1] if cs else None, cycle_s_median=round(statistics.median(cs), 1) if cs else None,
                    live_age_s=0.4)
        now = real_now
        fr = int(now * 2)  # new camera frame twice a second
        camera = dict(top=f"/img/top_{'ab'[fr % 2]}.jpg?v={fr}", wrist=f"/img/wrist_{'ab'[fr % 2]}.jpg?v={fr}", age_s=0.3)
        power = dict(live=False) if a.no_power else self.power(now, stage)
        if not a.no_power:
            n = len(sorted_rows)
            power["brick"] = dict(n=n, J=round(44.2 * 32.6, 0), Wh=round(44.2 * 32.6 / 3600, 3), s=32.6, W=44.2, domain="psys")

        if a.status == "idle":
            cell.update(status="idle", stage=None, stage_started=None, stage_elapsed=None, motion=None,
                        message="no cell program running")
            pipeline = [dict(p, state="done" if p["key"] == "optimize" else "idle", text=p["text"] if p["key"] == "reason" else None) for p in pipeline]
            camera = None
            th.append(dict(t=round(self.idle_at + 2, 2), kind="info", text="Run finished. Standing by: waiting for the next brick."))
            th = th[-30:]
        return dict(now=round(now, 3), demo=True, cell=cell, pipeline=pipeline, inspection=insp,
                    history=[{k2: v for k2, v in h.items() if k2 != "t"} for h in hist][-20:],
                    camera=camera, joints=self.joints(now) if a.status == "running" else None,
                    power=power, bench=BENCH, config=CONFIG, thoughts=th)


CONFIG = dict(act_device="GPU", anomaly_device="GPU",
              models=dict(pick="act-pick-v3", place_good="act-place-good-v3", place_bad="act-place-bad-v3"),
              anomaly_model="patchcore_views_384")


def _row(model, label, role, device, med, p95=None, first=None, comp=None, fps=None, pkg=None, gpu=None, mj=None, err=None, cont=False, prec="FP16"):
    return dict(model=model, label=label, role=role, device=device, median_ms=med, p95_ms=p95, first_ms=first, compile_s=comp,
                tput_fps=fps, pkg_W=pkg, gpu_W=gpu, npu_pct=0.0 if device != "NPU" else (None if err else 92.0),
                mJ_above_idle=mj, precision=prec, weights="act-pick-v3" if role == "policy" else label, error=err, contended=cont)


BENCH = dict(
    stamp="2026-11-15 02:10", started="2026-11-15T02:04:31", idle_W=6.4, idle_psys_W=21.8,
    system=dict(openvino="2025.3.0-19807-44526285f24", devices=dict(CPU="Intel(R) Core(TM) Ultra X9 388H", GPU="Intel(R) Graphics (iGPU)", NPU="Intel(R) AI Boost")),
    rows=[
        _row("ACT policy", "act-pick-v3", "policy", "GPU", 28.4, 33.1, 412, 21.5, 41.0, 24.1, 9.2, 410),
        _row("ACT policy", "act-pick-v3", "policy", "NPU", 41.2, 44.0, 96, 38.0, 25.5, 14.9, 0.4, 330),
        _row("ACT policy", "act-pick-v3", "policy", "CPU", 95.3, 121.0, 140, 2.1, 12.8, 31.0, 0.3, 2350),
        _row("PatchCore", "patchcore_views_384", "anomaly", "GPU", 265.0, 288.0, 1650, 34.0, 4.1, 27.2, 10.8, 5100),
        _row("PatchCore", "patchcore_views_384", "anomaly", "NPU", None, err="compile failed: memory bank (359 MB) too large for the NPU"),
        _row("PatchCore", "patchcore_views_384", "anomaly", "CPU", 3900.0, 4210.0, 4400, 3.2, 0.26, 33.5, 0.3, 98000, cont=True),
        _row("EfficientAd", "efficientad_256", "anomaly", "GPU", 6.1, 7.0, 220, 12.0, 190.0, 25.0, 10.5, 95),
        _row("EfficientAd", "efficientad_256", "anomaly", "NPU", 8.7, 9.4, 60, 29.0, 130.0, 13.2, 0.5, 70),
        _row("EfficientAd", "efficientad_256", "anomaly", "CPU", 120.0, 141.0, 180, 1.5, 9.0, 32.0, 0.3, 3100),
    ],
)


def make_handler(sim: Sim):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str, cache: str = "no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                with open(os.path.join(HERE, "index.html"), "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8")
            if path == "/_frame":  # dev aid: the page at an exact CSS size, scaled to fit, e.g. /_frame?w=1366&h=768
                q = dict(p.split("=", 1) for p in urlparse(self.path).query.split("&") if "=" in p)
                w, h = int(q.get("w", 1920)), int(q.get("h", 1080))
                html = (f"<!doctype html><body style='margin:0;background:#444;overflow:hidden'><iframe src='/' style='width:{w}px;height:{h}px;border:0;"
                        f"transform-origin:0 0' id=f></iframe><script>const s=()=>f.style.transform='scale('+Math.min(innerWidth/{w},innerHeight/{h})+')';"
                        f"addEventListener('resize',s);s()</script></body>")
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if path == "/_set":  # dev aid: /_set?freeze=good_real&status=idle&no_power=1&blank=0 (empty freeze = rotate)
                q = dict(p.split("=", 1) for p in urlparse(self.path).query.split("&") if "=" in p)
                if "freeze" in q:
                    sim.a.freeze = q["freeze"] if q["freeze"] in SCEN else None
                    sim._plans, sim._starts = [], [sim.t0]
                if "status" in q:
                    sim.a.status = q["status"]
                for k in ("no_power", "blank"):
                    if k in q:
                        setattr(sim.a, k, q[k] == "1")
                return self._send(200, json.dumps(vars(sim.a)).encode(), "application/json")
            if path == "/api/state":
                return self._send(200, json.dumps(sim.state()).encode(), "application/json")
            if path.startswith("/img/"):
                rel = os.path.normpath(path[5:]).replace("\\", "/")
                full = os.path.join(IMG, rel)
                if rel.startswith("..") or os.path.isabs(rel) or not os.path.isfile(full):
                    return self._send(404, b"not found", "text/plain")
                with open(full, "rb") as fh:
                    return self._send(200, fh.read(), mimetypes.guess_type(full)[0] or "image/jpeg", "max-age=3600")
            return self._send(404, b"not found", "text/plain")

    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--freeze", choices=sorted(SCEN), help="always inspect this brick")
    ap.add_argument("--status", choices=["running", "idle"], default="running")
    ap.add_argument("--no-power", action="store_true")
    ap.add_argument("--blank", action="store_true")
    ap.add_argument("--dump", help="write one state snapshot to this path and exit")
    a = ap.parse_args()
    sim = Sim(a)
    if a.dump:
        sim.t0 = time.time() - 15.0 - 7.4  # mid-way through the first live cycle: picking
        st = sim.state()
        with open(a.dump, "w", encoding="utf-8") as fh:
            json.dump(st, fh, indent=1, ensure_ascii=False)
        print("wrote", a.dump)
        return 0
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(sim))
    print(f"mock dashboard on http://127.0.0.1:{a.port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
