#!/usr/bin/env python3
"""What the cell sees on the table, for the jury dashboard's top-camera overlay.

Reads the top-camera JPEG that tools/cell_live.py publishes (/dev/shm/checkurai-live/top.jpg); never opens a camera.
Bowls, markers and target zones come from the bowl work, read-only (tools/bowl_check.py status(), owned by that work):
  - pick plate (yellow), good bowl (lilac), defect bowl (orange): where each is now (c, r)
  - per bowl, where its policy saw it in its training demos (target circle = the demos' median spot) and the zone of
    all demo positions, mapped into the live picture through the ArUco table markers; on_target = the bowl centre is
    within 15 px (~1.7 cm) of that median spot (bowl_check's "ok" since Sun 02:40: on the arm the pick policy went
    where the plate usually stood, not where it was, so "inside the zone" was not good enough). The zone is drawing only.
  - camera_moved_px: how far the camera has moved since that policy's demos (undone in software per model)
Added here (not from the bowl work):
  - bricks waiting on the pick plate (compact non-yellow colour blobs inside the plate circle)
  - the inspection spot, approximate (tools/bowl_follow.spot_xy in the canonical picture, mapped to live)
Everything is in top-camera pixels (640x480) so the page can draw it with an SVG viewBox over the picture.
While the arm moves it hides markers and bowls: the last good bowl position is held (held=True), and the
in-zone verdict is only updated while the arm is at HOME (verdict_t says when it was measured).

    python tools/scene_overlay.py [image.jpg]      # print one result (default: the live top.jpg)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIVE = Path(os.environ.get("CHECKURAI_LIVE_DIR", "/dev/shm/checkurai-live"))
LABELS = {"yellow": "Pick plate", "lilac": "Good bowl", "orange": "Defect bowl"}
STAGES = {"yellow": "pick", "lilac": "place_good", "orange": "place_bad"}
CANON = ROOT / "config" / "cell_ref" / "top_place_demo.png"
BRICK_PX = 1300  # area of one 2x2 Duplo brick top in the 640x480 top camera (measured 1130-1530 px on the cell)


def _pt(p) -> list[float]:
    return [round(float(p[0]), 1), round(float(p[1]), 1)]


def bricks_on_plate(img: np.ndarray, c, r: float) -> list[dict]:
    """Bricks waiting on the yellow pick plate: compact non-yellow colour blobs inside the plate circle.
    Plate H 23-25 / V 170-200; a lime brick is H 30-33 / V 90-150, so the yellow range is kept narrow. A yellow
    brick on the yellow plate is not seen. Tuned on two top-camera frames (1 and 3 bricks)."""
    import cv2

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    disk = np.zeros((h, w), np.uint8)
    cv2.circle(disk, (int(c[0]), int(c[1])), int(r * 0.90), 255, -1)
    yellow = cv2.inRange(hsv, (17, 150, 150), (28, 255, 255))
    colour = cv2.bitwise_or(cv2.inRange(hsv, (0, 70, 40), (180, 255, 255)), cv2.inRange(hsv, (0, 0, 0), (180, 255, 70)))
    m = cv2.bitwise_and(cv2.bitwise_and(colour, cv2.bitwise_not(yellow)), disk)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for k in range(1, n):
        x, y, bw, bh, a = (int(v) for v in st[k])
        merged = a > 2200  # touching bricks form one blob: estimate how many from the area of one brick
        if a < 600 or a / float(bw * bh) < (0.3 if merged else 0.45) or (not merged and max(bw, bh) > 2.2 * min(bw, bh)):
            continue  # a Duplo brick from the top camera is ~1000-1500 px and compact; the rim's shadow is a sliver
        out.append({"c": _pt(cen[k]), "box": [x, y, bw, bh], "n": max(1, round(a / BRICK_PX)), "merged": merged})
    return out


def brick_count(bricks: list[dict]) -> dict:
    """{n, approx}: approx when touching bricks were counted from their blob's area."""
    return {"n": int(sum(b.get("n", 1) for b in bricks)), "approx": any(b.get("merged") for b in bricks)}


class Scene:
    def __init__(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        import bowl_check as BC  # noqa: PLC0415 - owned by the bowl work, used read-only
        import policy_tune as PT  # noqa: PLC0415

        self.PT, self.BC = PT, BC
        self.targets = BC.Targets()
        self.seen: deque = deque(maxlen=5)  # marker detections of the last frames (median against occlusion)
        self.held: dict[str, dict] = {}     # colour -> last time the bowl was seen
        self.verdict: dict[str, dict] = {}  # colour -> last in-zone verdict measured with the arm at HOME
        self.canon = None
        self.spot_canon = None

    def _spot(self, live: dict) -> list[float] | None:
        """The inspection spot (approximate): bowl_follow.spot_xy in the canonical picture, mapped to live."""
        import cv2

        if self.spot_canon is None:
            import bowl_follow  # noqa: PLC0415

            self.canon = self.PT.detect(cv2.imread(str(CANON)))
            self.spot_canon = np.asarray(bowl_follow.spot_xy(self.canon), np.float64)
        hi = np.linalg.inv(np.array(self.PT.fit_warp(live, self.canon)["H"]))
        return _pt(cv2.perspectiveTransform(self.spot_canon.reshape(1, 1, 2), hi)[0, 0])

    def analyse(self, img: np.ndarray, at_home: bool = True) -> dict[str, Any]:
        PT, BC = self.PT, self.BC
        h, w = img.shape[:2]
        now = time.time()
        try:
            self.seen.append(PT.detect(img))
        except SystemExit:  # not a 640x480 frame
            pass
        live = PT.median_markers(list(self.seen), 0.5) if self.seen else {}
        st = BC.status(img, live, self.targets.get()) if live else []
        by = {s["color"]: s for s in st}
        objs = []
        for color in ("yellow", "lilac", "orange"):
            s = dict(by.get(color) or {"color": color, "c": None})
            o: dict[str, Any] = {"color": color, "label": LABELS[color], "stage": s.get("stage") or STAGES[color],
                                 "model": s.get("model"), "target_c": s.get("target_c"), "target_r": s.get("target_r"),
                                 "zone": s.get("zone"), "camera_moved_px": s.get("camera_moved_px")}
            if s.get("c") is None:  # find the bowl even without markers (no target then)
                fb = PT.find_bowl(img, color)
                if fb is not None:
                    s["c"], s["r"] = _pt(fb[0]), round(float(fb[1]), 1)
            if s.get("c") is not None:
                self.held[color] = {"c": s["c"], "r": s["r"], "t": now}
                o.update(c=s["c"], r=s["r"], held=False)
                if at_home and "ok" in s and s.get("target_c") is not None:
                    off_cm = s.get("off_cm")
                    if off_cm is None and s.get("off_px") is not None:  # bowl_check's estimate: bowl radius ~9 cm
                        off_cm = round(float(s["off_px"]) / max(float(s["r"]), 40.0) * 9.0, 1)
                    self.verdict[color] = {"ok": bool(s["ok"]), "off_px": s.get("off_px"), "off_cm": off_cm,
                                           "arrow_to": s.get("arrow_to"), "msg": s.get("msg"), "t": round(now, 2)}
            elif color in self.held and now - self.held[color]["t"] < 120:
                o.update(c=self.held[color]["c"], r=self.held[color]["r"], held=True)
            v = self.verdict.get(color)
            if v:
                o.update(on_target=v["ok"], in_zone=v["ok"], off_px=v["off_px"], off_cm=v["off_cm"], arrow_to=v["arrow_to"],
                         verdict_msg=v["msg"], verdict_t=v["t"])
            if color == "yellow" and o.get("c") is not None and not o.get("held"):
                o["bricks"] = bricks_on_plate(img, o["c"], o["r"])
                o["brick_count"] = brick_count(o["bricks"])
            objs.append(o)
        spot = None
        if live:
            try:
                spot = self._spot(live)
            except (SystemExit, Exception):  # noqa: BLE001 - too few shared markers: no spot this time
                spot = None
        moved = [o["camera_moved_px"] for o in objs if o.get("camera_moved_px") is not None]
        return {"t": round(now, 2), "size": [w, h], "objects": objs, "spot": spot, "at_home": at_home,
                "camera_moved_px": round(float(max(moved)), 1) if moved else None,
                "markers": {"seen": len(live), "ids": sorted(int(i) for i in live),
                            "centres": [_pt(np.asarray(c).mean(axis=0)) for _, c in sorted(live.items())]}}


class SceneWatcher:
    """Re-analyses the live top.jpg at most every `every_s` seconds, in a background thread, when it changed.
    at_home(): whether the arm is at HOME now (the in-zone verdict is only measured then)."""

    def __init__(self, every_s: float = 0.5, at_home: Callable[[], bool] | None = None) -> None:
        self.every_s = every_s
        self.at_home = at_home or (lambda: True)
        self.latest: dict | None = None
        self.error: str | None = None
        self._mtime = 0.0
        self._scene: Scene | None = None
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True, name="scene-overlay").start()

    def _loop(self) -> None:
        import cv2

        while not self._stop.wait(self.every_s):
            p = LIVE / "top.jpg"
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m == self._mtime:
                continue
            self._mtime = m
            try:
                if self._scene is None:
                    self._scene = Scene()
                img = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                t0 = time.perf_counter()
                res = self._scene.analyse(img, bool(self.at_home()))
                res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
                res["frame_t"] = round(m, 2)
                self.latest, self.error = res, None
            except BaseException:  # noqa: BLE001 - incl. SystemExit from the bowl code: keep the dashboard up
                self.error = traceback.format_exc(limit=2)[-300:]
                self._stop.wait(5)

    def get(self, max_age_s: float = 3.0) -> dict | None:
        r = self.latest
        if r is None or time.time() - r.get("frame_t", 0) > max_age_s:
            return None
        return r


def main() -> int:
    import cv2

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else LIVE / "top.jpg"
    img = cv2.imread(str(src))
    if img is None:
        print(f"cannot read {src}")
        return 1
    sc = Scene()
    t0 = time.perf_counter()
    res = sc.analyse(img)
    res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
