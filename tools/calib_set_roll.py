"""Apply calib_pair.ROLL (wrist-roll ranges) to the current calibration: servos, LeRobot files, HOME roll.
Only valid after calib_pair.py (both wrists homed straight at 2047). Then run tools/studio_push_calib.py.

    python tools/calib_set_roll.py
"""
import json
import shutil
import sys
import time

from calib_pair import FILES, HOME_FILE, PORTS, ROLL, ROLL_TURN, ROOT, TPD, as_dict, norm
from lerobot.motors import MotorCalibration
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

stamp = time.strftime("%m%d-%H%M%S")
backup = ROOT / f"calib-backup-{stamp}-roll"
backup.mkdir(parents=True)
for p in [*FILES.values(), HOME_FILE]:
    shutil.copy2(p, backup / p.name)
cal = {a: json.loads(p.read_text()) for a, p in FILES.items()}
for a in cal:
    print(f"{a}: wrist_roll {cal[a]['wrist_roll']['range_min']}..{cal[a]['wrist_roll']['range_max']} -> {ROLL[a][0]}..{ROLL[a][1]}")
    cal[a]["wrist_roll"]["range_min"], cal[a]["wrist_roll"]["range_max"] = ROLL[a]

buses = {"leader": SOLeader(SOLeaderTeleopConfig(port=PORTS["leader"], id="checkurai_leader")).bus,
         "follower": SOFollower(SOFollowerRobotConfig(port=PORTS["follower"], id="checkurai_follower")).bus}
try:
    for a, b in buses.items():
        b.connect()
        b.disable_torque()
        b.write_calibration({j: MotorCalibration(**c) for j, c in cal[a].items()})
        if as_dict(b.read_calibration()) != cal[a]:
            sys.exit(f"{a}: servos did not keep the values")
finally:
    for b in buses.values():
        if b.is_connected:
            b.disconnect(disable_torque=True)
for a, p in FILES.items():
    p.write_text(json.dumps(cal[a], indent=4))

h = json.loads(HOME_FILE.read_text())
i = h["joints"].index("wrist_roll")
before = h["home"][i]
h["home"][i] = round(norm("wrist_roll", 2047 + ROLL_TURN * TPD, cal["follower"]["wrist_roll"]), 3)
h["source"]["note"] += f" Wrist roll re-ranged {stamp}: follower {ROLL['follower']}, leader {ROLL['leader']}."
HOME_FILE.write_text(json.dumps(h, indent=1))
print(f"HOME wrist_roll {before} -> {h['home'][i]}   backup: {backup}")
