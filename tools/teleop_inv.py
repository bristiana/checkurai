"""CheckurAI teleop v3: follower mirrors leader 1:1 (joints in invert.json reversed).
Start: slow 4 s ramp to the leader pose. Speed limit acts on the COMMAND (max STEP per tick), not on the
gap to the follower's present position, so joints under gravity load are never starved. Ctrl+C to stop."""
import json, os, time
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
HERE = os.path.dirname(os.path.abspath(__file__))
INV = {j + ".pos" for j in json.load(open(os.path.join(HERE, "invert.json")))}
L = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00"
F = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00"
RAMP_S, HZ, STEP = 4.0, 60, 6.0          # STEP: max change per tick (deg; gripper %) -> 360 deg/s
LAG = os.path.expanduser(time.strftime("~/checkurai/logs/lag-%H%M%S.csv"))
leader = SOLeader(SOLeaderTeleopConfig(port=L, id="checkurai_leader"))
follower = SOFollower(SOFollowerRobotConfig(port=F, id="checkurai_follower", max_relative_target=None))
def lead():
    a = leader.get_action()
    for k in INV: a[k] = -a[k]
    return a
def slew(prev, tgt):
    return {k: prev[k] + max(-STEP, min(STEP, tgt[k] - prev[k])) for k in tgt}
leader.connect(calibrate=False); follower.connect(calibrate=False)
lagf = open(LAG, "w")
try:
    for attempt in range(10):            # the bus can miss the very first packet right after connect
        try:
            obs = follower.get_observation(); a = lead(); break
        except ConnectionError as e:
            print(f"read failed ({e.__class__.__name__}), retry {attempt + 1}/10", flush=True); time.sleep(0.3)
    else:
        raise SystemExit("The follower or leader does not answer: check the 12 V power and the servo cables.")
    start = {k: float(obs[k]) for k in a}; keys = list(a)
    lagf.write("t," + ",".join(f"{k[:-4]}_cmd,{k[:-4]}_act" for k in keys) + "\n")
    print("Start differences (leader - follower):", {k[:-4]: round(a[k] - start[k], 1) for k in a}, flush=True)
    print(f"Ramping follower to the leader pose over {RAMP_S:.0f} s: hold the leader still...", flush=True)
    cmd, t0 = dict(start), time.time()
    while (t := time.time() - t0) < RAMP_S:
        a = lead(); al = t / RAMP_S
        cmd = slew(cmd, {k: start[k] + (a[k] - start[k]) * al for k in a}); follower.send_action(cmd)
        time.sleep(1 / HZ)
    print("Teleop running 1:1. Reversed joints:", sorted(k[:-4] for k in INV) or "none", "| Ctrl+C to stop (follower goes limp).", flush=True)
    n, t0 = 0, time.time()
    while True:
        cmd = slew(cmd, lead()); follower.send_action(cmd); n += 1
        if n % 30 == 0:                      # every 0.5 s: how far the real arm is from the command
            o = follower.get_observation()
            lagf.write(f"{time.time()-t0:.2f}," + ",".join(f"{cmd[k]:.1f},{o[k]:.1f}" for k in keys) + "\n"); lagf.flush()
            print(f"\r{n/(time.time()-t0):5.1f} Hz | gap cmd-real: " + "  ".join(f"{k[:-4]}={cmd[k]-o[k]:+5.1f}" for k in keys), end="", flush=True)
        time.sleep(1 / HZ)
except KeyboardInterrupt:
    pass
finally:
    lagf.close(); follower.disconnect(); leader.disconnect(); print(f"\nStopped, torque off. Lag log: {LAG}")
