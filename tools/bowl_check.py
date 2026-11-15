"""Where must the bowls stand? Each policy learned its bowl where it stood in that policy's own demos.

The pick policy takes the brick from the yellow plate, place_good drops it in the lilac bowl, place_bad in the
orange bowl (models named in config/cell.json). For each bowl, tools/policy_tune.py measured where it stood
in every training demo of its policy (median position + how much it varied, cached next to the export). The
ArUco markers locate that spot in the live top camera, so the target stays right even if the camera moved:
dashed circle + cross = the median bowl position of the demos; outline = the zone of all bowl-centre positions in
its demos; solid circle + x = where it is now; arrow = the way back onto the dashed circle. "OK" = the bowl centre
is within 15 px (~1.7 cm) of the demos' median. On the arm (Sun 02:40) the pick policy went where the plate usually
stood (missed the brick by ~2 cm with the plate 21 px off, though inside the zone), so the zone is not enough:
the policies only partly follow a moved plate / bowl (tools/bowl_react.py: ~63 % for a bowl).

    ~/physical-ai-studio/application/backend/.venv/bin/python tools/bowl_check.py     # -> logs/bowl_check.jpg
Live in the browser: tools/aim_view.py shows the same as its third panel.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy_tune as PT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BOWLS = (("pick_model", "yellow"), ("place_good_model", "lilac"), ("place_bad_model", "orange"))
DRAW = {"lilac": (230, 150, 200), "orange": (0, 120, 255), "yellow": (0, 220, 255)}
OK_PX = 15.0  # ~1.7 cm: the policy follows the bowl a little, and the bowl is ~18 cm wide
find_bowl = PT.find_bowl


class Targets:
    """The bowl targets in each policy's training view (re-read when config/cell.json changes)."""

    def __init__(self):
        self.stamp, self.items = None, []

    def get(self) -> list[dict]:
        st = (ROOT / "config" / "cell.json").stat().st_mtime
        if st != self.stamp:
            cfg = PT.cfg()
            items = []
            for key, color in BOWLS:
                name = cfg.get(key)
                try:
                    ref = PT.train_top_ref(PT.model_record(name))
                    b = (ref.get("bowls") or {}).get(color)
                    items.append({"model": name, "color": color, "stage": key[:-6], "ref": PT.ref_corners(ref), "bowl": b,
                                  "err": None if b else "bowl not found in its training demos "
                                                        "(delete top_ref.json next to the export to re-measure)"})
                except (SystemExit, Exception) as e:  # noqa: BLE001 - show it on the page
                    items.append({"model": name, "color": color, "stage": key[:-6], "err": str(e)[:160]})
            self.stamp, self.items = st, items
        return self.items


def nearest_on(poly: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The point of a polygon (or segment / single point) closest to p."""
    if len(poly) == 1:
        return poly[0]
    best, bd = poly[0], 1e18
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
        q = a + t * ab
        dq = float(np.hypot(*(q - p)))
        if dq < bd:
            best, bd = q, dq
    return best


def status(live_img: np.ndarray, live: dict, targets: list[dict]) -> list[dict]:
    """Per bowl, in live top-camera pixels (JSON-ready): the demos' median spot (target_c, target_r), the zone of
    all bowl centres in its demos (zone, polygon), where the bowl is now (c, r; None if not seen), ok, off_px
    (distance to the demos' median), arrow_to (that median, when it is off by more than OK_PX), camera_moved_px, msg."""
    out = []
    for t in targets:
        s = {"color": t["color"], "stage": t.get("stage"), "model": t["model"], "ok": False, "c": None}
        out.append(s)
        if t.get("err"):
            s["msg"] = t["err"]
            continue
        try:
            w = PT.fit_warp(live, t["ref"])  # live -> training view
        except SystemExit as e:
            s["msg"] = str(e)
            continue
        hi = np.linalg.inv(np.array(w["H"]))
        c, r = np.array(t["bowl"]["c"]), t["bowl"]["r"]
        edge = cv2.perspectiveTransform(np.array([[c, c + [r, 0], c + [0, r]]], np.float64), hi)[0]
        tc = edge[0]
        tr = float(np.mean([np.linalg.norm(edge[1] - tc), np.linalg.norm(edge[2] - tc)]))
        # the zone the policy knows: every bowl-centre position in its demos (it follows the bowl inside it)
        hull = t["bowl"].get("hull")
        if hull and len(hull) >= 2:
            zone = cv2.perspectiveTransform(np.array([hull], np.float64), hi)[0]
        else:  # older cache: a circle of the demos' spread
            tol = max(OK_PX, t["bowl"]["spread_p75_px"])
            ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            zone = tc + tol * np.c_[np.cos(ang), np.sin(ang)]
        s.update(target_c=[round(float(v), 1) for v in tc], target_r=round(tr, 1),
                 zone=np.round(zone, 1).tolist(), camera_moved_px=w["moved_px"], warp=w["kind"])
        now = find_bowl(live_img, t["color"])
        if now is None:
            s["msg"] = f"bowl not visible now - put it in the dashed {t['color']} circle"
            continue
        nc, nr = now
        s.update(c=[round(float(v), 1) for v in nc], r=round(float(nr), 1))
        target = tc  # where it usually stood in the demos: the policies go there more than they follow the bowl
        d = target - nc
        off = float(np.hypot(*d))
        s["off_px"] = round(off, 1)
        if off <= OK_PX:
            s.update(ok=True, msg=f"OK - {off:.0f} px (~{off / max(nr, 40) * 9.0:.0f} cm) from where it stood "
                                  "in its demos")
            continue
        side = []
        if abs(d[0]) > 3:
            side.append(f"{'right' if d[0] > 0 else 'left'} {abs(d[0]):.0f}")
        if abs(d[1]) > 3:
            side.append(f"{'down' if d[1] > 0 else 'up'} {abs(d[1]):.0f}")
        cm = off / max(nr, 40) * 9.0  # bowl radius ~9 cm -> rough distance on the table
        s.update(arrow_to=[round(float(v), 1) for v in target], off_cm=round(cm, 1),
                 msg=f"~{cm:.0f} cm from where it stood in its demos - slide it along the arrow "
                     f"(in the picture: {', '.join(side)} px) until its centre (x) is on the dashed circle's cross")
    return out


def check(live_img: np.ndarray, live: dict, targets: list[dict]) -> tuple[np.ndarray, list[str], bool]:
    """Draw targets / current bowls on the live picture; returns (picture, lines, all ok)."""
    img = live_img.copy()
    st = status(live_img, live, targets)
    for s in st:
        col = DRAW[s["color"]]
        if "target_c" in s:
            tc, tr = (int(s["target_c"][0]), int(s["target_c"][1])), int(s["target_r"])
            for a in range(0, 360, 20):  # dashed target circle
                cv2.ellipse(img, tc, (tr, tr), 0, a, a + 10, col, 3)
            cv2.polylines(img, [np.array(s["zone"]).astype(np.int32)], True, col, 2)
            cv2.drawMarker(img, tc, col, cv2.MARKER_CROSS, 18, 2)
        if s["c"] is not None:
            nc = (int(s["c"][0]), int(s["c"][1]))
            cv2.circle(img, nc, int(s["r"]), col, 2)
            cv2.drawMarker(img, nc, col, cv2.MARKER_TILTED_CROSS, 14, 2)
            if "arrow_to" in s:
                cv2.arrowedLine(img, nc, (int(s["arrow_to"][0]), int(s["arrow_to"][1])), col, 4, tipLength=0.3)
    lines = [f"{s['color']} ({s['model']}): {s['msg']}" for s in st]
    return img, lines, all(s["ok"] for s in st)


def main() -> int:
    import cam_lock as L

    L.open_source("top")
    fr = L.frames("top", 6, gap=0.05)
    live = PT.median_markers([PT.detect(f) for f in fr], 0.5)
    img, lines, ok = check(fr[-1], live, Targets().get())
    out = ROOT / "logs" / "bowl_check.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), img)
    print("\n".join(lines))
    print(f"{'ALL BOWLS OK' if ok else 'move the bowls as above'} - picture: {out} ({time.strftime('%H:%M:%S')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
