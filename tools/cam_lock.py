"""Freeze exposure, white balance and focus on the cell cameras (V4L2), so the picture stays identical
between demo recording, anomaly training and the judged run.

Most UVC cameras don't report the values their auto mode picked, so `lock` measures the live picture
(through the running camera dashboard, tools/camdash.py on :8080) and searches for fixed values:
  exposure  -> same mean brightness as under auto (and at most ~1.5 % blown-out pixels)
  white bal -> the white paper comes out neutral
  focus     -> sharpest picture (only cameras with a focus motor, i.e. the inspection webcam)
Then it saves everything to config/cam_lock.json.

    python tools/cam_lock.py lock [role ...]   # roles: top wrist inspection (default: all)
    python tools/cam_lock.py apply             # re-apply the saved values (after a replug / reboot)
    python tools/cam_lock.py auto  [role ...]  # back to auto exposure / WB / focus
    python tools/cam_lock.py show

Keep hands and the arm still while `lock` runs (about 30 s per camera).
"""
import fcntl
import json
import math
import os
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

URL = "http://127.0.0.1:8080"
HERE = Path(__file__).resolve().parent
CFG = HERE.parent / "config/cam_lock.json"
PAUSE = HERE.parent / "config/.cam_lock_running"  # tells tools/cam_guard.py to keep its hands off
ROLES = ("top", "wrist", "inspection")
EXP_MAX = 330  # 100 us units -> 33 ms, keeps 30 fps possible
SETTLE = 0.8

VIDIOC_QUERYCTRL, VIDIOC_G_CTRL, VIDIOC_S_CTRL = 0xC0445624, 0xC008561B, 0xC008561C
NEXT = 0x80000000


class V4L:
    def __init__(self, dev):
        self.dev, self.fd, self.c = dev, os.open(dev, os.O_RDWR), {}
        cid = NEXT
        while True:
            buf = bytearray(struct.pack("II32siiiiIII", cid, 0, b"", 0, 0, 0, 0, 0, 0, 0))
            try:
                fcntl.ioctl(self.fd, VIDIOC_QUERYCTRL, buf)
            except OSError:
                break
            i, typ, name, mn, mx, step, dflt, flags = struct.unpack("II32siiiiI", bytes(buf[:60]))
            cid = i | NEXT
            if typ == 6:
                continue
            key = name.split(b"\0")[0].decode().lower().replace(" ", "_").replace(",", "")
            self.c[key] = dict(id=i, min=mn, max=mx, step=step, default=dflt)

    def has(self, k):
        return k in self.c and self.get(k) is not None

    def get(self, k):
        g = bytearray(struct.pack("Ii", self.c[k]["id"], 0))
        try:
            fcntl.ioctl(self.fd, VIDIOC_G_CTRL, g)
        except OSError:
            return None
        return struct.unpack("Ii", g)[1]

    def set(self, k, v):
        c = self.c[k]
        v = int(round(max(c["min"], min(c["max"], v))))
        if c["step"] > 1:
            v = c["min"] + round((v - c["min"]) / c["step"]) * c["step"]
        fcntl.ioctl(self.fd, VIDIOC_S_CTRL, bytearray(struct.pack("Ii", c["id"], v)))
        return v


def devices():
    out = subprocess.run([sys.executable, str(HERE / "find_cams.py"), "--json"], capture_output=True, text=True).stdout
    return json.loads(out)


SHARED = {}  # role -> physicalai SharedCamera (used when Studio holds the camera; needs Studio's venv)


def studio_camera(role):
    """Attach to (or start) Studio's shared publisher for this camera, built the way Studio builds it."""
    from physicalai.capture import ColorMode, SharedCamera
    from physicalai.config import Config
    api = "http://127.0.0.1:7860/api/projects"
    pid = json.load(urllib.request.urlopen(api, timeout=5))[0]["id"]
    cam = next(c for c in json.load(urllib.request.urlopen(f"{api}/{pid}/cameras", timeout=5)) if c["name"] == role)
    args = {k: cam["payload"][k] for k in ("width", "height", "fps") if cam["payload"].get(k)}
    if cam["driver"] == "usb_camera":
        cls, args["device"] = "physicalai.capture.UVCCamera", cam["fingerprint"]
    else:
        cls, args["serial_number"] = "physicalai.capture.RealSenseCamera", cam["fingerprint"]["serial"]
    sc = SharedCamera(camera=Config(cls, args), color_mode=ColorMode.RGB, validate_on_connect=False,
                      overwrite_settings=False, idle_timeout=5.0)
    sc.connect(timeout=10)
    return sc


def open_source(role):
    try:
        urllib.request.urlopen(f"{URL}/frame/{role}", timeout=3).read()
        return "camdash"
    except Exception:
        SHARED[role] = studio_camera(role)
        return "Studio shared camera"


def frames(role, n=3, gap=0.08):
    res = []
    for _ in range(n):
        if role in SHARED:
            for _ in range(2):  # drop queued frames
                f = np.asarray(SHARED[role].read(timeout=3).data)
            f = cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2BGR)  # publisher sends RGB
        else:
            buf = urllib.request.urlopen(f"{URL}/frame/{role}", timeout=3).read()
            f = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if f is not None:
            res.append(f)
        time.sleep(gap)
    return res


def stats(fs):
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fs]).astype(np.float32)
    f = fs[-1]
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    lum = g[-1]
    paper = (lum >= np.percentile(lum, 60)) & (hsv[..., 1] < 50) & (lum < 250)
    if paper.mean() < 0.02:
        paper = lum >= np.percentile(lum, 70)
    b, gr, r = (float(f[..., i][paper].mean()) for i in range(3))
    return dict(mean=float(g.mean()), clip=float(100 * (g[-1] >= 250).mean()),
                rg=r / gr, bg=b / gr, sharp=float(cv2.Laplacian(lum, cv2.CV_32F).var()),
                motion=float(np.abs(np.diff(g, axis=0)).mean()) if len(fs) > 1 else 0.0)


def measure(role, n=3):
    time.sleep(SETTLE)
    frames(role, 2, 0.03)  # flush
    return stats(frames(role, n))


def lock_exposure(role, v, base):
    v.set("auto_exposure", 1)  # manual
    if v.has("exposure_dynamic_framerate"):
        v.set("exposure_dynamic_framerate", 0)
    lo, hi = max(1, v.c["exposure_time_absolute"]["min"]), min(EXP_MAX, v.c["exposure_time_absolute"]["max"])
    target, best = base["mean"], None
    for _ in range(9):
        e = math.exp((math.log(lo) + math.log(hi)) / 2)
        e = v.set("exposure_time_absolute", e)
        s = measure(role, 2)
        if best is None or abs(s["mean"] - target) < abs(best[1]["mean"] - target):
            best = (e, s)
        print(f"    exposure {e:5d} -> brightness {s['mean']:5.1f} (target {target:5.1f})  clipped {s['clip']:4.1f}%")
        if abs(s["mean"] - target) < 2:
            break
        lo, hi = (e, hi) if s["mean"] < target else (lo, e)
        if hi - lo < 2:
            break
    e, s = best
    # the wrist cam sees the lamp itself at HOME; don't darken its whole picture for that
    while role != "wrist" and s["clip"] > 1.5 and e > 3:
        e = v.set("exposure_time_absolute", e * 0.88)
        s = measure(role, 2)
        print(f"    exposure {e:5d} -> brightness {s['mean']:5.1f}  clipped {s['clip']:4.1f}%  (cutting blown-out highlights)")
    v.set("exposure_time_absolute", e)
    return e


def lock_wb(role, v):
    v.set("white_balance_automatic", 0)
    c = v.c["white_balance_temperature"]
    lo, hi, best = c["min"], c["max"], None
    for _ in range(9):
        t = v.set("white_balance_temperature", (lo + hi) / 2)
        s = measure(role, 2)
        cast = math.log(s["rg"]) - math.log(s["bg"])  # >0 warm, <0 cold
        if best is None or abs(cast) < abs(best[1]):
            best = (t, cast, s)
        print(f"    white balance {t:5d} K -> paper R/G {s['rg']:4.2f}  B/G {s['bg']:4.2f}")
        if abs(cast) < 0.02:
            break
        lo, hi = (lo, t) if cast > 0 else (t, hi)  # higher setting = warmer picture
        if hi - lo <= max(2, c["step"]):
            break
    return v.set("white_balance_temperature", best[0])


def lock_focus(role, v):
    v.set("focus_automatic_continuous", 0)
    c = v.c["focus_absolute"]
    step = max(c["step"], 1)
    scores = {}

    def score(f):
        f = v.set("focus_absolute", f)
        if f not in scores:
            time.sleep(0.4)
            scores[f] = measure(role, 2)["sharp"]
        return f, scores[f]

    for f in range(c["min"], min(c["max"], 700) + 1, 50):
        f, s = score(f)
        print(f"    focus {f:4d} -> sharpness {s:6.1f}")
    best = max(scores, key=scores.get)
    for f in range(max(c["min"], best - 40), min(c["max"], best + 40) + 1, max(10, step)):
        score(f)
    best = max(scores, key=scores.get)
    print(f"    focus -> {best} (sharpness {scores[best]:.1f})")
    return v.set("focus_absolute", best)


def read_lock():
    return json.loads(CFG.read_text()) if CFG.exists() else {}


def cmd_lock(roles):
    PAUSE.touch()
    try:
        _lock(roles)
    finally:
        PAUSE.unlink(missing_ok=True)


def _lock(roles):
    devs, saved = devices(), read_lock()
    for role in roles:
        dev = devs.get(role)
        if not dev:
            print(f"\n## {role}: camera not found, skipped"); continue
        v = V4L(dev)
        print(f"\n## {role}  {dev}  (frames via {open_source(role)})")
        original = {k: v.get(k) for k in ("auto_exposure", "exposure_time_absolute", "exposure_dynamic_framerate",
                                          "white_balance_automatic", "white_balance_temperature",
                                          "focus_automatic_continuous", "focus_absolute", "gain",
                                          "power_line_frequency") if k in v.c}
        if original.get("auto_exposure") != 3 or original.get("white_balance_automatic") != 1:
            print("  (not on auto now - switching auto on for 4 s to get the reference picture)")
            v.set("auto_exposure", 3); v.set("white_balance_automatic", 1)
            if "focus_automatic_continuous" in v.c:
                v.set("focus_automatic_continuous", 1)
            time.sleep(4)
        base = measure(role, 5)
        print(f"  auto picture: brightness {base['mean']:.1f}  clipped {base['clip']:.1f}%  paper R/G {base['rg']:.2f} "
              f"B/G {base['bg']:.2f}  sharpness {base['sharp']:.0f}  motion {base['motion']:.1f}")
        if base["motion"] > 4:
            print("  !! the picture is moving (arm or hands in view?) - result may be off; run again when still")
        ctl = {}
        if "power_line_frequency" in v.c and v.c["power_line_frequency"]["max"] >= 1:
            ctl["power_line_frequency"] = v.set("power_line_frequency", 1)
        ctl["gain"] = v.get("gain") if "gain" in v.c else None
        ctl["exposure_time_absolute"] = lock_exposure(role, v, base)
        ctl["auto_exposure"] = 1
        if "exposure_dynamic_framerate" in v.c and v.get("exposure_dynamic_framerate") is not None:
            ctl["exposure_dynamic_framerate"] = 0
        if role == "wrist":  # manual WB on this camera gives a green cast at every temperature
            ctl["white_balance_automatic"] = v.set("white_balance_automatic", 1)
        else:
            ctl["white_balance_temperature"] = lock_wb(role, v)
            ctl["white_balance_automatic"] = 0
        if "focus_absolute" in v.c and "focus_automatic_continuous" in v.c:
            ctl["focus_absolute"] = lock_focus(role, v)
            ctl["focus_automatic_continuous"] = 0
        after = measure(role, 5)
        snap = HERE.parent / "snapshots/lock"
        snap.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snap / f"{role}.jpg"), frames(role, 1)[0])
        print(f"  LOCKED:brightness {after['mean']:.1f}  clipped {after['clip']:.1f}%  paper R/G {after['rg']:.2f} "
              f"B/G {after['bg']:.2f}  sharpness {after['sharp']:.0f}")
        print(f"  values: {ctl}")
        saved[role] = dict(controls={k: x for k, x in ctl.items() if x is not None}, original=original,
                           auto_picture=base, locked_picture=after, device_name=Path(f"/sys/class/video4linux/{Path(dev).name}/name").read_text().strip(),
                           when=time.strftime("%Y-%m-%d %H:%M:%S"))
        CFG.parent.mkdir(exist_ok=True)
        CFG.write_text(json.dumps(saved, indent=1))
    for sc in SHARED.values():
        sc.disconnect()
    print(f"\nsaved {CFG}")


ORDER = ("power_line_frequency", "auto_exposure", "exposure_dynamic_framerate", "exposure_time_absolute", "gain",
         "white_balance_automatic", "white_balance_temperature", "focus_automatic_continuous", "focus_absolute")


def cmd_apply():
    devs, saved = devices(), read_lock()
    for role, d in saved.items():
        dev = devs.get(role)
        if not dev:
            print(f"{role}: camera not found"); continue
        v = V4L(dev)
        done = {k: v.set(k, d["controls"][k]) for k in ORDER if k in d["controls"] and k in v.c}
        bad = {k: (x, v.get(k)) for k, x in done.items() if v.get(k) != x}
        print(f"{role} {dev}: {done}" + (f"  !! did not stick: {bad}" if bad else "  ok"))


def cmd_auto(roles):
    devs = devices()
    for role in roles:
        if role in devs:
            v = V4L(devs[role])
            v.set("auto_exposure", 3); v.set("white_balance_automatic", 1)
            if "focus_automatic_continuous" in v.c:
                v.set("focus_automatic_continuous", 1)
            print(role, "back on auto")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    roles = [r for r in sys.argv[2:] if r in ROLES] or list(ROLES)
    if cmd == "lock":
        cmd_lock(roles)
    elif cmd == "apply":
        cmd_apply()
    elif cmd == "auto":
        cmd_auto(roles)
    else:
        print(json.dumps({r: d["controls"] for r, d in read_lock().items()}, indent=1))
