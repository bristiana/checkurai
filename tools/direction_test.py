"""DIRECTION TEST. Both arms limp. Hold one arm in each hand and move ONE joint at a time,
the SAME physical way on both arms (same pose, same direction). Each joint gets a verdict:
SAME = fine, OPPOSITE = that joint is reversed between the arms. ENTER to finish."""
import json, math, os, select, sys, time
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
L = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00"
F = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00"
OUT = os.path.expanduser("~/checkurai/logs/direction-test.json")
leader = SOLeader(SOLeaderTeleopConfig(port=L, id="checkurai_leader"))
follower = SOFollower(SOFollowerRobotConfig(port=F, id="checkurai_follower"))
leader.connect(calibrate=False); follower.connect(calibrate=False)
follower.bus.disable_torque()
phase = {"leader": {m: int(leader.bus.read("Phase", m, normalize=False)) for m in leader.bus.motors},
         "follower": {m: int(follower.bus.read("Phase", m, normalize=False)) for m in follower.bus.motors}}
J = list(leader.bus.motors)
st = {j: {"ll": 0.0, "ff": 0.0, "lf": 0.0, "ml": 0.0, "mf": 0.0} for j in J}
def verdict(s):
    if s["ml"] < 12 or s["mf"] < 12: return "move this joint on BOTH arms", None
    c = s["lf"] / math.sqrt(s["ll"] * s["ff"] + 1e-9)
    return ("SAME  (ok)" if c > 0.4 else "OPPOSITE  <-- reversed" if c < -0.4 else "unclear, move again"), round(c, 2)
pa = po = None
try:
    while True:
        a = leader.get_action(); o = follower.get_observation()
        if pa is not None:
            for j in J:
                dl, df = a[j + ".pos"] - pa[j + ".pos"], o[j + ".pos"] - po[j + ".pos"]
                if abs(dl) < 0.4: dl = 0.0
                if abs(df) < 0.4: df = 0.0
                s = st[j]; s["ll"] += dl * dl; s["ff"] += df * df; s["lf"] += dl * df; s["ml"] += abs(dl); s["mf"] += abs(df)
        pa, po = a, o
        out = ["\033[H\033[J=== DIRECTION TEST: one joint at a time, move it the SAME way on BOTH arms ===",
               f"{'joint':14s} {'LEADER':>8s} {'FOLLOWER':>9s}   verdict"]
        for j in J:
            out.append(f"{j:14s} {a[j+'.pos']:8.1f} {o[j+'.pos']:9.1f}   {verdict(st[j])[0]}")
        out.append("\nPress ENTER when every joint has a verdict.")
        print("\n".join(out), flush=True)
        if select.select([sys.stdin], [], [], 0.05)[0]:
            sys.stdin.readline(); break
finally:
    res = {"verdicts": {j: verdict(st[j]) for j in J}, "phase_registers": phase, "time": time.strftime("%H:%M:%S")}
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nSaved to", OUT); print(json.dumps(res["verdicts"], indent=1))
    leader.disconnect(); follower.bus.disconnect(disable_torque=True)
