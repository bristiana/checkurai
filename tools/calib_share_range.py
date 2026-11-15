"""Give leader and follower the SAME range on some joints, so they read the same at the same pose.

Only valid right after tools/calib_pair.py, which homes both arms in the same middle pose (2047 = middle).
The shared range is the overlap of both recorded ranges, so the follower never gets a wider limit than
it reached. Writes the servos and the LeRobot files and converts the HOME pose (same physical pose).
Then run tools/studio_push_calib.py.

    python tools/calib_share_range.py shoulder_pan [more joints]
"""
import json
import shutil
import sys
import time

from calib_pair import FILES, HOME_FILE, PORTS, ROOT, as_dict, norm, to_raw
from lerobot.motors import MotorCalibration
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

joints = sys.argv[1:]
if not joints or "wrist_roll" in joints or "gripper" in joints:
    sys.exit("usage: calib_share_range.py <body joints...> (not wrist_roll or gripper)")
stamp = time.strftime("%m%d-%H%M%S")
backup = ROOT / f"calib-backup-{stamp}-share"
backup.mkdir(parents=True)
for p in [*FILES.values(), HOME_FILE]:
    shutil.copy2(p, backup / p.name)
cal = {a: json.loads(p.read_text()) for a, p in FILES.items()}
old_f = json.loads(json.dumps(cal["follower"]))
for j in joints:
    if cal["leader"][j]["homing_offset"] == cal["follower"][j]["homing_offset"] == 0:
        sys.exit(f"{j}: no homing found; run calib_pair.py first")
    lo = max(cal["leader"][j]["range_min"], cal["follower"][j]["range_min"])
    hi = min(cal["leader"][j]["range_max"], cal["follower"][j]["range_max"])
    print(f"{j}: leader {cal['leader'][j]['range_min']}..{cal['leader'][j]['range_max']}, "
          f"follower {cal['follower'][j]['range_min']}..{cal['follower'][j]['range_max']} -> both {lo}..{hi}")
    for a in cal:
        cal[a][j]["range_min"], cal[a][j]["range_max"] = lo, hi

buses = {"leader": SOLeader(SOLeaderTeleopConfig(port=PORTS["leader"], id="checkurai_leader")).bus,
         "follower": SOFollower(SOFollowerRobotConfig(port=PORTS["follower"], id="checkurai_follower")).bus}
try:
    for a, b in buses.items():
        b.connect()
        b.disable_torque()
        new = {j: MotorCalibration(**c) for j, c in cal[a].items()}
        b.write_calibration(new)
        back = as_dict(b.read_calibration())
        if back != cal[a]:
            sys.exit(f"{a}: servos did not keep the values: {back}")
finally:
    for b in buses.values():
        if b.is_connected:
            b.disconnect(disable_torque=True)
for a, p in FILES.items():
    p.write_text(json.dumps(cal[a], indent=4))

h = json.loads(HOME_FILE.read_text())
prev = list(h["home"])
for i, j in enumerate(h["joints"]):
    if j in joints:   # same homing, so the same raw position; only the range changed
        h["home"][i] = round(norm(j, to_raw(j, prev[i], old_f[j]), cal["follower"][j]), 3)
h["source"]["note"] += f" Shared leader/follower range on {', '.join(joints)} ({stamp})."
HOME_FILE.write_text(json.dumps(h, indent=1))
print("HOME before:", prev)
print("HOME now:   ", h["home"])
print("backup:", backup)
