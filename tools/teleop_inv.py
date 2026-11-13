"""CheckurAI teleop: follower mirrors leader; joints in invert.json are reversed.
Starts with a slow 4 s ramp from the follower's current pose to the leader's pose. Ctrl+C to stop."""
import json, os, time
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
HERE = os.path.dirname(os.path.abspath(__file__))
INV = {j + ".pos" for j in json.load(open(os.path.join(HERE, "invert.json")))}
L = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00"
F = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00"
RAMP_S, HZ = 4.0, 60
leader = SOLeader(SOLeaderTeleopConfig(port=L, id="checkurai_leader"))
follower = SOFollower(SOFollowerRobotConfig(port=F, id="checkurai_follower", max_relative_target=5.0))
def lead():
    a = leader.get_action()
    for k in INV: a[k] = -a[k]
    return a
leader.connect(calibrate=False); follower.connect(calibrate=False)
try:
    obs = follower.get_observation(); a = lead()
    start = {k: float(obs[k]) for k in a}
    print("Start differences (leader - follower):", {k[:-4]: round(a[k] - start[k], 1) for k in a}, flush=True)
    print(f"Ramping follower to the leader pose over {RAMP_S:.0f} s: hold the leader still...", flush=True)
    t0 = time.time()
    while (t := time.time() - t0) < RAMP_S:
        a = lead(); al = t / RAMP_S
        follower.send_action({k: start[k] + (a[k] - start[k]) * al for k in a})
        time.sleep(1 / HZ)
    print("Teleop running. Reversed joints:", sorted(k[:-4] for k in INV), "| Ctrl+C to stop (follower goes limp).", flush=True)
    n, t0 = 0, time.time()
    while True:
        a = lead(); follower.send_action(a); n += 1
        if n % 120 == 0:
            print(f"\r{n/(time.time()-t0):5.1f} Hz  " + "  ".join(f"{k[:-4]}={v:6.1f}" for k, v in a.items()), end="", flush=True)
        time.sleep(1 / HZ)
except KeyboardInterrupt:
    pass
finally:
    follower.disconnect(); leader.disconnect(); print("\nStopped, torque off.")
