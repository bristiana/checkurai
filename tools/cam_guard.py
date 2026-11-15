"""Keep the cell cameras locked: every few seconds, re-apply config/cam_lock.json to any camera whose settings
differ (a replugged UVC camera comes back with factory auto settings), and keep Studio's USB camera
indexes pointing at the right /dev/videoN (they change on replug).

Only touches V4L2 controls (no streaming), so it is safe while Studio or the dashboard use the cameras.
Pauses while `cam_lock.py lock` runs.

    python tools/cam_guard.py            # loop forever, log in logs/cam_guard.log
    python tools/cam_guard.py --once     # one pass, print what it did
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cam_lock as L  # noqa: E402

API = "http://127.0.0.1:7860/api/projects"
LOG = Path(__file__).resolve().parent.parent / "logs/cam_guard.log"
EVERY = 5.0
_last_msg = {}


def log(msg, key=None):
    if key is not None and _last_msg.get(key) == msg:
        return  # don't repeat the same complaint every 5 s
    if key is not None:
        _last_msg[key] = msg
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def enforce_lock(devs):
    lock = L.read_lock()
    for role, d in lock.items():
        dev = devs.get(role)
        if not dev:
            log(f"{role}: camera not found", key=role)
            continue
        v = L.V4L(dev)
        try:
            ctl = d["controls"]
            diff = {k: (v.get(k), x) for k, x in ctl.items() if k in v.c and v.get(k) != x}
            if diff:
                for k in L.ORDER:
                    if k in ctl and k in v.c:
                        v.set(k, ctl[k])
                still = {k: v.get(k) for k in ctl if k in v.c and v.get(k) != ctl[k]}
                log(f"{role} {dev}: re-applied lock (was {diff})" + (f"  !! not sticking: {still}" if still else ""))
            else:
                log(f"{role}: ok on {dev}", key=role)
        finally:
            os.close(v.fd)


def studio_sync(devs):
    try:
        pid = json.load(urllib.request.urlopen(API, timeout=3))[0]["id"]
        cams = json.load(urllib.request.urlopen(f"{API}/{pid}/cameras", timeout=3))
    except Exception as e:
        log(f"Studio not reachable ({type(e).__name__})", key="studio")
        return
    _last_msg.pop("studio", None)
    for c in cams:
        dev = devs.get(c["name"])
        if c["driver"] != "usb_camera" or not dev:
            continue
        idx = int(dev.replace("/dev/video", ""))
        if c["fingerprint"].get("index") != idx:
            old = c["fingerprint"].get("index")
            c["fingerprint"]["index"] = idx
            body = {k: c[k] for k in ("id", "driver", "name", "fingerprint", "hardware_name", "payload")}
            req = urllib.request.Request(f"{API}/{pid}/cameras/{c['id']}", data=json.dumps(body).encode(),
                                         method="PUT", headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
            log(f"Studio camera '{c['name']}' index {old} -> {idx}")


def one_pass():
    if L.PAUSE.exists() and time.time() - L.PAUSE.stat().st_mtime < 900:
        log("paused: cam_lock.py lock is running", key="pause")
        return
    _last_msg.pop("pause", None)
    devs = L.devices()
    enforce_lock(devs)
    studio_sync(devs)


if __name__ == "__main__":
    LOG.parent.mkdir(exist_ok=True)
    if "--once" in sys.argv:
        one_pass()
        sys.exit(0)
    log("cam_guard started")
    while True:
        try:
            one_pass()
        except Exception as e:
            log(f"error: {type(e).__name__}: {e}", key="err")
        time.sleep(EVERY)
