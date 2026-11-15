"""Camera health check through the running camera dashboard (tools/camdash.py on :8080).

For each camera: fps, brightness, clipped highlights / crushed shadows, sharpness, colour cast,
flicker (frame-to-frame brightness wobble and horizontal banding), plus the V4L2 auto/locked controls.
Saves one frame per camera to ~/checkurai/snapshots/test/<role>.jpg.

    python tools/camtest.py [seconds]
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

URL = "http://127.0.0.1:8080"
OUT = Path.home() / "checkurai/snapshots/test"
OUT.mkdir(parents=True, exist_ok=True)
SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
HERE = Path(__file__).resolve().parent
KEEP = ("auto_exposure", "exposure_time_absolute", "exposure_dynamic_framerate", "white_balance_automatic",
        "white_balance_temperature", "focus_automatic_continuous", "focus_absolute", "power_line_frequency",
        "gain", "brightness", "backlight_compensation")


def controls(dev: str) -> dict:
    out = subprocess.run([sys.executable, str(HERE / "v4lctl.py"), dev], capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        name = line[:40].strip().lower().replace(" ", "_").replace(",", "")
        if name in KEEP and "value=" in line:
            res[name] = line.rsplit("value=", 1)[1].split()[0]
    return res


devs = json.loads(subprocess.run([sys.executable, str(HERE / "find_cams.py"), "--json"], capture_output=True, text=True).stdout)
stats = json.load(urllib.request.urlopen(URL + "/stats", timeout=5))
for role in ("top", "wrist", "inspection"):
    if role not in stats:
        print(f"\n## {role}: NOT AVAILABLE ({devs.get(role, 'no device')})")
        continue
    frames, t0 = [], time.time()
    while time.time() - t0 < SECS:
        try:
            buf = urllib.request.urlopen(f"{URL}/frame/{role}", timeout=3).read()
        except Exception as e:
            print(role, "frame error", e); break
        f = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if f is not None:
            frames.append(f)
        time.sleep(0.03)
    if not frames:
        print(f"\n## {role}: no frames"); continue
    f = frames[-1]
    cv2.imwrite(str(OUT / f"{role}.jpg"), f)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    means = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).mean() for x in frames])
    rows = g.mean(axis=1)
    band = float(np.std(rows - cv2.GaussianBlur(rows.reshape(-1, 1), (1, 31), 0).ravel()))
    b, gr, r = (f[..., i].mean() for i in range(3))
    s = json.load(urllib.request.urlopen(URL + "/stats", timeout=5)).get(role, {})
    print(f"\n## {role}  {devs.get(role)}  {s.get('size')}  {s.get('fps')} fps  ({len(frames)} frames grabbed)")
    print(f"  brightness mean {g.mean():5.1f} /255   clipped(>=250) {100 * (g >= 250).mean():4.1f}%   crushed(<=8) {100 * (g <= 8).mean():4.1f}%")
    print(f"  sharpness (Laplacian var) {cv2.Laplacian(g, cv2.CV_32F).var():7.1f}")
    print(f"  colour R/G {r / gr:4.2f}  B/G {b / gr:4.2f}   (1.00 = neutral)")
    print(f"  flicker: frame-to-frame brightness std {means.std():4.2f}   row banding {band:4.2f}")
    if devs.get(role):
        print("  controls:", controls(devs[role]))
print(f"\nframes saved in {OUT}")
