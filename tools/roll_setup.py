"""Set the follower's wrist-roll HOME by hand and give the wrist the range it can really turn.

    bash ~/checkurai/roll_setup.sh     (follower limp: hold the arm)

  A. Turn the follower wrist to where it should sit at HOME (leader wrist straight = this). ENTER:
     the follower wrist_roll is re-homed there (it reads 2047 = the middle of the encoder).
  B. Turn it gently both ways until it touches something or the cable is tight. ENTER.
Then both arms get the same wrist_roll range around their middle (the leader's middle is its straight
wrist from calib_pair.py), 10 deg clear of what B found and at most +-170 deg (the encoder wraps at 180).
Writes the servos, the LeRobot files and the HOME roll. Then run tools/studio_push_calib.py.
"""
import json
import select
import shutil
import sys
import time

from calib_pair import FILES, HOME_FILE, PORTS, ROOT, TPD, as_dict, norm
from lerobot.motors import MotorCalibration
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

MARGIN, CAP = 10.0, 170.0


def key():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline()
    return None


def show(lines):
    sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n"); sys.stdout.flush()


stamp = time.strftime("%m%d-%H%M%S")
backup = ROOT / f"calib-backup-{stamp}-rollsetup"
backup.mkdir(parents=True)
for p in [*FILES.values(), HOME_FILE]:
    shutil.copy2(p, backup / p.name)
cal = {a: json.loads(p.read_text()) for a, p in FILES.items()}
fol = SOFollower(SOFollowerRobotConfig(port=PORTS["follower"], id="checkurai_follower")).bus
lea = SOLeader(SOLeaderTeleopConfig(port=PORTS["leader"], id="checkurai_leader")).bus
fol.connect(); fol.disable_torque()
old = fol.read_calibration()["wrist_roll"]
raw = lambda: int(fol.read("Present_Position", "wrist_roll", normalize=False))
done = False
try:
    # A. HOME by hand. Angle shown from the calib_pair 'straight' (2047), unwrapped across the encoder edge.
    prev, turns = raw(), 0
    while True:
        r = raw()
        if r - prev > 2048: turns -= 1
        elif prev - r > 2048: turns += 1
        prev = r
        ang = (r + 4096 * turns - 2047) / TPD
        show(["=== WRIST HOME, step A ===  (follower limp, hold the arm)", "",
              "Turn the FOLLOWER wrist by hand to where it should sit at HOME",
              "(camera NOT underneath; you chose about 180 deg from straight).", "",
              f"  now: {ang:+7.1f} deg from straight", "", "Hold it there and press ENTER."])
        if key() is not None:
            break
        time.sleep(0.05)
    home_from_straight = ang
    homing = fol.set_half_turn_homings(["wrist_roll"])["wrist_roll"]

    # B. range around the new HOME
    prev, turns, lo, hi = raw(), 0, 0.0, 0.0
    while True:
        r = raw()
        if r - prev > 2048: turns -= 1
        elif prev - r > 2048: turns += 1
        prev = r
        ang = (r + 4096 * turns - 2047) / TPD
        lo, hi = min(lo, ang), max(hi, ang)
        show(["=== WRIST RANGE, step B ===  (follower limp)", "",
              "Turn the follower wrist gently BOTH ways from HOME until it touches something",
              "or the cable gets tight. Do not force it.", "",
              f"  now: {ang:+7.1f} deg from HOME      furthest: {lo:+7.1f} .. {hi:+7.1f}", "",
              "Press ENTER when done."])
        if key() is not None:
            break
        time.sleep(0.05)
    a, b = max(lo + MARGIN, -CAP), min(hi - MARGIN, CAP)
    if b - a < 90:
        raise SystemExit(f"only {b - a:.0f} deg of usable wrist travel ({lo:+.0f}..{hi:+.0f}); not changing anything")
    rng = (round(2047 + a * TPD), round(2047 + b * TPD))

    cal["follower"]["wrist_roll"].update(homing_offset=int(homing), range_min=rng[0], range_max=rng[1])
    cal["leader"]["wrist_roll"].update(range_min=rng[0], range_max=rng[1])
    fol.write_calibration({j: MotorCalibration(**c) for j, c in cal["follower"].items()})
    lea.connect(); lea.disable_torque()
    lea.write_calibration({j: MotorCalibration(**c) for j, c in cal["leader"].items()})
    for a_, bus in (("follower", fol), ("leader", lea)):
        if as_dict(bus.read_calibration()) != cal[a_]:
            raise SystemExit(f"{a_}: servos did not keep the values")
    for a_, p in FILES.items():
        p.write_text(json.dumps(cal[a_], indent=4))
    done = True

    h = json.loads(HOME_FILE.read_text())
    i = h["joints"].index("wrist_roll")
    h["home"][i] = round(norm("wrist_roll", 2047, cal["follower"]["wrist_roll"]), 3)
    h["source"]["note"] += (f" Wrist roll set up by hand {stamp}: follower HOME = {home_from_straight:+.0f} deg from straight "
                            f"(re-homed there), both arms wrist_roll {rng[0]}..{rng[1]} ({a:+.0f}..{b:+.0f} deg).")
    HOME_FILE.write_text(json.dumps(h, indent=1))
    msg = (f"HOME at {home_from_straight:+.1f} deg from straight; free {lo:+.1f}..{hi:+.1f}; "
           f"range {a:+.0f}..{b:+.0f} deg = ticks {rng}; HOME roll {h['home'][i]}")
    print("\n" + msg)
    (ROOT / "logs" / "roll_setup.txt").open("a").write(f"{stamp} {msg}\n")
except BaseException as e:
    if not done:
        fol.disable_torque()
        fol.write_calibration({"wrist_roll": old}, cache=False)
        print(f"\n{e if isinstance(e, SystemExit) else 'Stopped.'}  Wrist roll put back; nothing saved.")
        (ROOT / "logs" / "roll_setup.txt").open("a").write(f"{stamp} ABORTED {e!r}\n")
    if not isinstance(e, (KeyboardInterrupt, SystemExit)):
        raise
finally:
    for bus in (fol, lea):
        if bus.is_connected:
            bus.disconnect(disable_torque=True)
