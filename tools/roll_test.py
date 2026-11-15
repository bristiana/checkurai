"""Find how far the FOLLOWER wrist can roll by hand (follower limp). Shows the angle from 'straight'
(the pose used in calib_pair.py step 1) and the furthest it reached each way. ENTER to finish."""
import select
import sys
import time

from calib_pair import PORTS
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

bus = SOFollower(SOFollowerRobotConfig(port=PORTS["follower"], id="checkurai_follower")).bus
bus.connect()
bus.disable_torque()
lo = hi = None
try:
    while True:
        raw = int(bus.read("Present_Position", "wrist_roll", normalize=False))
        ang = (raw - 2047) * 360 / 4095
        lo = ang if lo is None else min(lo, ang)
        hi = ang if hi is None else max(hi, ang)
        sys.stdout.write("\033[H\033[J=== FOLLOWER WRIST ROLL TEST (follower limp, hold the arm) ===\n\n"
                         "Turn the follower's WRIST by hand, gently, both ways, until it touches something\n"
                         "(camera holder, cable, gripper) or the cable pulls tight. Do not force it.\n\n"
                         f"  now: {ang:+7.1f} deg from straight      raw {raw}\n"
                         f"  furthest:  {lo:+7.1f}  ..  {hi:+7.1f}\n\n"
                         "Press ENTER when done.\n")
        sys.stdout.flush()
        if select.select([sys.stdin], [], [], 0.1)[0]:
            sys.stdin.readline(); break
finally:
    print(f"RESULT wrist_roll range from straight: {lo:+.1f} .. {hi:+.1f} deg")
    with open("/home/intel/checkurai/logs/roll_test.txt", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {lo:+.1f} {hi:+.1f}\n")
    bus.disconnect(disable_torque=True)
