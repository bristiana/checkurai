"""Cell freeze check: did a camera, the stage or the light move since the reference was saved?

Uses the ArUco markers (DICT_4X4_50: 0-3 on the table corners, 4-5 on the stage), the whole-image shift
against a reference frame, the brightness, and the locked camera settings (config/cam_lock.json).
Frames come from the camera dashboard if it runs, otherwise from Studio's shared camera feed
(run with Studio's venv then, see cam_lock.py).

    python tools/cell_check.py save      # freeze: store the reference (arm at HOME, hands out, no bricks moving)
    python tools/cell_check.py           # compare now vs the reference -> PASS / what moved
    python tools/cell_check.py watch     # repeat every 2 s (re-aim a bumped camera until it says PASS)
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cam_lock as L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / "config/cell_ref"
REF = ROOT / "config/cell_ref.json"
ROLES = ("top", "inspection", "wrist")
MAX_MARKER_PX = 4.0   # marker centre drift that counts as "moved"
MAX_SHIFT_PX = 3.0    # whole-image shift that counts as "moved"
MAX_LIGHT = 0.08      # relative brightness change that counts as "light changed"
DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                              cv2.aruco.DetectorParameters())


def markers(gray):
    corners, ids, _ = DET.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): dict(c=[float(x) for x in c[0].mean(0)], side=float(np.mean(
        [np.linalg.norm(c[0][k] - c[0][(k + 1) % 4]) for k in range(4)]))) for i, c in zip(ids.ravel(), corners)}


def grab(role):
    if role not in L.SHARED:
        L.open_source(role)
    fs = L.frames(role, 4)
    f = fs[-1]
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return f, g, L.stats(fs)


def lock_ok(role, lock, devs):
    if role not in lock or role not in devs:
        return None
    v = L.V4L(devs[role])
    bad = {k: (x, v.get(k)) for k, x in lock[role]["controls"].items() if k in v.c and v.get(k) != x}
    return bad


def save():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    ref = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "cams": {}}
    for role in ROLES:
        try:
            f, g, s = grab(role)
        except Exception as e:
            print(f"{role}: no frames ({e})"); continue
        cv2.imwrite(str(REF_DIR / f"{role}.png"), f)
        m = markers(g)
        ref["cams"][role] = dict(markers=m, mean=s["mean"], clip=s["clip"], motion=s["motion"])
        print(f"{role:10s} markers {sorted(m)}  brightness {s['mean']:.1f}  clipped {s['clip']:.1f}%  motion {s['motion']:.1f}"
              + ("   !! picture moving - save again when still" if s["motion"] > 4 else ""))
    REF.write_text(json.dumps(ref, indent=1))
    print(f"reference saved: {REF} + {REF_DIR}/")


def shift(ref_gray, g):
    a, b = (np.float32(x) for x in (ref_gray, g))
    (dx, dy), resp = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    return dx, dy, resp


def check(quiet=False):
    ref = json.loads(REF.read_text())
    lock = json.loads(L.CFG.read_text()) if L.CFG.exists() else {}
    devs = L.devices()
    all_ok = True
    for role, r in ref["cams"].items():
        problems = []
        try:
            f, g, s = grab(role)
        except Exception as e:
            print(f"{role:10s} FAIL  no frames ({e})"); all_ok = False; continue
        bad = lock_ok(role, lock, devs)
        if bad:
            problems.append(f"camera settings changed {bad} -> run: python tools/cam_lock.py apply")
        d = s["mean"] / max(r["mean"], 1) - 1
        if abs(d) > MAX_LIGHT:
            problems.append(f"light {'brighter' if d > 0 else 'darker'} by {100 * abs(d):.0f}% (lamp moved / room light?)")
        now = markers(g)
        if role != "wrist":  # the wrist moves with the arm; its markers only mean something at HOME
            for mid, m in r["markers"].items():
                mid = int(mid)
                if mid not in now:
                    problems.append(f"marker {mid} not seen (covered, moved out of view, or camera moved)")
                    continue
                dx, dy = (now[mid]["c"][k] - m["c"][k] for k in (0, 1))
                if np.hypot(dx, dy) > MAX_MARKER_PX:
                    problems.append(f"marker {mid} moved {dx:+.0f},{dy:+.0f} px (scale x{now[mid]['side'] / m['side']:.2f})")
            refimg = cv2.imread(str(REF_DIR / f"{role}.png"), cv2.IMREAD_GRAYSCALE)
            if refimg is not None and refimg.shape == g.shape:
                dx, dy, resp = shift(refimg, g)
                if np.hypot(dx, dy) > MAX_SHIFT_PX and resp > 0.05:
                    problems.append(f"whole picture shifted {dx:+.1f},{dy:+.1f} px")
        if s["motion"] > 4:
            problems.append("picture is moving (hands/arm in view?) - check again when still")
        all_ok &= not problems
        print(f"{role:10s} {'PASS' if not problems else 'FAIL'}  markers {sorted(now)}  brightness {s['mean']:.0f} (ref {r['mean']:.0f})")
        for p in problems:
            print("           -", p)
    print("CELL OK" if all_ok else "CELL CHANGED - fix the items above (or, if the change is intended and NO demos exist yet: relock + save)")
    return all_ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    try:
        if cmd == "save":
            save()
        elif cmd == "watch":
            while True:
                print(time.strftime("\n%H:%M:%S")); check()
                time.sleep(2)
        else:
            sys.exit(0 if check() else 1)
    finally:
        for sc in L.SHARED.values():
            sc.disconnect()
