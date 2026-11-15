"""Print both arms' joint positions in Studio units next to HOME (config/home_pose.json). Read-only, torque untouched.
Needs both serial ports free (no teleop, no Studio environment loaded).

    python tools/arm_pose.py
"""
import json
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PORTS = {"leader": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00",
         "follower": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00"}


def read(port):
    bus = FeetechMotorsBus(port, {n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if n == "gripper"
                                           else MotorNormMode.RANGE_M100_100) for i, n in enumerate(NAMES)})
    bus.connect(handshake=False)
    try:
        bus.calibration = bus.read_calibration()
        return bus.sync_read("Present_Position")
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    home = json.loads((ROOT / "config/home_pose.json").read_text())["home"]
    pos = {k: read(p) for k, p in PORTS.items()}
    worst = 0.0
    print(f"{'joint':14s} {'leader':>8s} {'follower':>9s} {'gap':>6s} {'HOME':>8s}")
    for n, h in zip(NAMES, home):
        gap = abs(pos["leader"][n] - pos["follower"][n])
        worst = max(worst, gap)
        print(f"{n:14s} {pos['leader'][n]:8.1f} {pos['follower'][n]:9.1f} {gap:6.1f} {h:8.1f}")
    print(f"largest leader/follower gap: {worst:.1f}")
