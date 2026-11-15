"""Bring the follower to HOME (config/home_pose.json) over its own serial bus, slowly, if it is not there.

Needs the follower port free (no Studio session holding the arm). Moves in joint space with a smooth
start and stop, at most --speed Studio units per second on the joint that moves furthest, and leaves the
arm powered at HOME. Keep a hand near the 12 V switch.

    python tools/go_home.py            # move only if the arm is outside the HOME tolerance
    python tools/go_home.py --check    # only report
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FOLLOWER = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00"
RATE = 50  # Goal_Position updates per second


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report only, never move")
    ap.add_argument("--speed", type=float, default=25.0, help="Studio units per second (100 units = 90 deg)")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config" / "home_pose.json").read_text())
    home = np.array([dict(zip(cfg["joints"], cfg["home"]))[n] for n in NAMES])
    tol = np.array([dict(zip(cfg["joints"], cfg["tolerance"]))[n] for n in NAMES])

    bus = FeetechMotorsBus(FOLLOWER, {n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if n == "gripper"
                                               else MotorNormMode.RANGE_M100_100) for i, n in enumerate(NAMES)})
    bus.connect(handshake=False)
    try:
        for attempt in range(20):  # after a supply dip the servos restart: give them ~10 s to answer again
            try:
                bus.calibration = bus.read_calibration()
                break
            except ConnectionError as e:
                if attempt == 19:
                    raise
                if attempt == 0:
                    print(f"servo bus not answering yet ({str(e)[-40:]}); waiting up to 10 s ...", flush=True)
                time.sleep(0.5)
        read = lambda: np.array([bus.sync_read("Present_Position", num_retry=3)[n] for n in NAMES])
        q = read()
        dev = np.abs(q - home) / tol
        for n, a, h, d in zip(NAMES, q, home, dev):
            print(f"  {n:14s} now {a:8.1f}  HOME {h:8.1f}  {'ok' if d <= 1 else 'OFF'}")
        if dev.max() <= 1:
            print("follower is at HOME")
            return 0
        if args.check:
            print("follower is NOT at HOME")
            return 1
        dist = float(np.abs(home - q).max())
        secs = max(dist / args.speed, 1.5)
        print(f"moving the follower to HOME over {secs:.1f} s ...", flush=True)
        bus.sync_write("Goal_Position", dict(zip(NAMES, q)))  # hold where it is before the torque comes on
        bus.enable_torque()
        steps = int(secs * RATE)
        t0 = time.perf_counter()
        for i in range(1, steps + 1):
            s = i / steps
            s = s * s * (3 - 2 * s)  # smooth start and stop
            bus.sync_write("Goal_Position", dict(zip(NAMES, q + (home - q) * s)))
            time.sleep(max(0.0, t0 + i / RATE - time.perf_counter()))
        time.sleep(1.0)
        q = read()
        dev = np.abs(q - home) / tol
        print(f"done; worst joint {NAMES[int(dev.argmax())]} at {dev.max():.2f} x tolerance "
              f"({'at HOME' if dev.max() <= 1 else 'still off HOME'})")
        return 0 if dev.max() <= 1 else 1
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    sys.exit(main())
