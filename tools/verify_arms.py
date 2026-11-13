"""Live side-by-side view of leader vs follower joint values. Both arms limp (no torque). ENTER to finish."""
import sys, select, time
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
L = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00"
F = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00"
import json, os
INV = {j + ".pos" for j in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "invert.json")))}
leader = SOLeader(SOLeaderTeleopConfig(port=L, id="checkurai_leader"))
follower = SOFollower(SOFollowerRobotConfig(port=F, id="checkurai_follower"))
leader.connect(calibrate=False); follower.connect(calibrate=False)
follower.bus.disable_torque()
try:
    while True:
        a = leader.get_action(); o = follower.get_observation()
        for k in INV: a[k] = -a[k]
        out = ["\033[H\033[J=== VERIFY: move BOTH arms into the SAME pose; numbers should match (±10) ===",
               f"{'joint':14s} {'LEADER':>8s} {'FOLLOWER':>9s} {'diff':>7s}"]
        for k in a:
            d = o[k] - a[k]
            flag = "  <-- check" if abs(d) > 15 else ""
            out.append(f"{k.replace('.pos',''):14s} {a[k]:8.1f} {o[k]:9.1f} {d:7.1f}{flag}")
        out.append("\n(gripper: 0 = closed, 100 = open; others in degrees)   Press ENTER when done.")
        print("\n".join(out), flush=True)
        if select.select([sys.stdin], [], [], 0.2)[0]:
            sys.stdin.readline(); break
finally:
    leader.disconnect(); follower.bus.disconnect(disable_torque=True)
