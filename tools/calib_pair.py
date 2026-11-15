"""Calibrate BOTH SO-101 arms (leader + follower) in one go, with live telemetry.

    bash ~/checkurai/final_calib.sh      # opens this, then teleop

Both arms are limp the whole time, so hold the follower. ENTER moves to the next step.
  1. MIDDLE  leader, then follower: put the arm in the middle pose, wrist roll straight. ENTER writes the homing.
  2. RANGES  move every joint of both arms to both hard stops (not the wrist roll). ENTER finishes.
  3. SAVE    writes the calibration to the servos and the LeRobot files, and converts the HOME pose.
  4. VERIFY  live leader vs follower in Studio units.
Ctrl+C before step 3 puts the old servo values back, and nothing is saved.

0 deg is the middle of each joint's recorded range, so each joint has to reach both of its hard stops.
Wrist roll is a full-turn joint with a fixed range. Measured with tools/roll_test.py (Sat 14 Nov): the
follower wrist is free from -80 deg (hard stop) to about +172 deg (stop) from straight. HOME is at -10 deg
(+90 and +170 put the camera underneath), so the follower gets -70..+160 deg and the leader the same range
plus 10 deg: leader straight = follower HOME.
"""
from __future__ import annotations

import json
import select
import shutil
import sys
import time
from pathlib import Path

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

ROOT = Path.home() / "checkurai"
CAL = Path.home() / ".cache/huggingface/lerobot/calibration"
FILES = {"leader": CAL / "teleoperators/so_leader/checkurai_leader.json",
         "follower": CAL / "robots/so_follower/checkurai_follower.json"}
PORTS = {"leader": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00",
         "follower": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00"}
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
RANGED = [j for j in JOINTS if j != "wrist_roll"]
TPD = 4095 / 360                                          # ticks per degree; homing puts "straight" at 2047
ROLL_FREE = (-70, 160)                                    # follower wrist, deg from straight (stops at -80 and ~+172)
ROLL_TURN = -10                                           # follower = leader - 10 deg: HOME = Marius's choice (camera not under)
ROLL = {"follower": tuple(round(2047 + d * TPD) for d in ROLL_FREE),
        "leader": tuple(round(2047 + (d - ROLL_TURN) * TPD) for d in ROLL_FREE)}
HOME_FILE = ROOT / "config/home_pose.json"
MID = 2047
DEG = 360 / 4095


def key() -> str | None:
    """The typed line if ENTER was pressed, else None (does not block)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip().lower()
    return None


def show(lines: list[str]) -> None:
    sys.stdout.write("\033[H" + "\n".join(l + "\033[K" for l in lines) + "\n\033[J")
    sys.stdout.flush()


def norm(j: str, raw: float, c: dict) -> float:
    """Studio units: -100..100 for the body joints, 0..100 (0 = closed) for the gripper."""
    lo, hi = c["range_min"], c["range_max"]
    f = (min(hi, max(lo, raw)) - lo) / (hi - lo)
    return f * 100 if j == "gripper" else f * 200 - 100


def to_raw(j: str, n: float, c: dict) -> float:
    lo, hi = c["range_min"], c["range_max"]
    return lo + (n / 100 if j == "gripper" else (n + 100) / 200) * (hi - lo)


def as_dict(cal: dict[str, MotorCalibration]) -> dict:
    return {j: {"id": c.id, "drive_mode": c.drive_mode, "homing_offset": int(c.homing_offset),
                "range_min": int(c.range_min), "range_max": int(c.range_max)} for j, c in cal.items()}


def read_all(buses) -> dict[str, dict[str, int]]:
    return {a: b.sync_read("Present_Position", normalize=False) for a, b in buses.items()}


def main() -> int:
    stamp = time.strftime("%m%d-%H%M%S")
    backup = ROOT / f"calib-backup-{stamp}"
    backup.mkdir(parents=True)
    old_files = {a: json.loads(p.read_text()) for a, p in FILES.items()}
    for a, p in FILES.items():
        shutil.copy2(p, backup / p.name)
    if HOME_FILE.exists():
        shutil.copy2(HOME_FILE, backup / HOME_FILE.name)

    buses = {"leader": SOLeader(SOLeaderTeleopConfig(port=PORTS["leader"], id="checkurai_leader", use_degrees=False)).bus,
             "follower": SOFollower(SOFollowerRobotConfig(port=PORTS["follower"], id="checkurai_follower", use_degrees=False)).bus}
    for b in buses.values():
        b.connect()
        b.disable_torque()
        for j in JOINTS:
            b.write("Operating_Mode", j, OperatingMode.POSITION.value)
    old_motor = {a: b.read_calibration() for a, b in buses.items()}
    (backup / "motors_before.json").write_text(json.dumps({a: as_dict(c) for a, c in old_motor.items()}, indent=2))
    print(f"backup: {backup}")

    saved = False
    try:
        # ---- 1. MIDDLE pose, one arm at a time --------------------------------------------------
        homings: dict[str, dict[str, int]] = {}
        for arm in ("leader", "follower"):
            while True:
                pos = read_all(buses)
                lines = [f"=== STEP 1/4  MIDDLE POSE: {arm.upper()} ===  (both arms limp; hold the follower)", "",
                         f"Put the {arm.upper()} in the middle pose:",
                         "  - base pointing straight forward",
                         "  - upper arm straight up, forearm pointing forward (elbow at about 90 deg)",
                         "  - wrist in line with the forearm",
                         "  - WRIST ROLL STRAIGHT (gripper upright, not turned): the same on both arms; this one matters most",
                         "  - gripper half open",
                         "Roughly right is fine for the big joints (within ~20 deg).", "",
                         "Degrees away from the LAST calibration's middle pose (aim for about 0):",
                         f"  {'joint':14s} {'LEADER':>8s} {'FOLLOWER':>9s}"]
                for j in JOINTS:
                    l = "   (set)" if arm == "follower" else f"{(pos['leader'][j] - MID) * DEG:+8.1f}"
                    lines.append(f"  {j:14s} {l:>8s} {(pos['follower'][j] - MID) * DEG:+9.1f}")
                lines += ["", f"Hold the {arm} still in the pose and press ENTER."]
                show(lines)
                if key() is not None:
                    a = buses[arm].sync_read("Present_Position", normalize=False)
                    time.sleep(0.3)
                    b = buses[arm].sync_read("Present_Position", normalize=False)
                    if max(abs(a[j] - b[j]) for j in JOINTS) > 8:
                        print("  ...it moved, hold it still and press ENTER again"); time.sleep(1.5); continue
                    homings[arm] = {j: int(v) for j, v in buses[arm].set_half_turn_homings().items()}
                    break
                time.sleep(0.1)
        for arm, b in buses.items():
            p = b.sync_read("Present_Position", normalize=False)
            print(f"{arm} after homing (all near 2047):", {j: int(p[j]) for j in JOINTS})

        # ---- 2. RANGES -----------------------------------------------------------------------------
        pos = read_all(buses)
        mins = {a: {j: pos[a][j] for j in RANGED} for a in buses}
        maxs = {a: dict(mins[a]) for a in buses}
        old_span = {a: {j: old_files[a][j]["range_max"] - old_files[a][j]["range_min"] for j in RANGED} for a in buses}
        msg = ""
        while True:
            pos = read_all(buses)
            for a in buses:
                for j in RANGED:
                    mins[a][j] = min(mins[a][j], pos[a][j]); maxs[a][j] = max(maxs[a][j], pos[a][j])
            span = {a: {j: maxs[a][j] - mins[a][j] for j in RANGED} for a in buses}
            lines = ["=== STEP 2/4  RANGES ===  move EVERY joint of BOTH arms slowly to BOTH hard stops", "",
                     "  - not the wrist roll (it stays straight; its range is fixed)",
                     "  - gripper: fully closed and fully open (leader trigger too)",
                     "  - follower: mind the wrist camera and its cable", "",
                     f"  {'joint':14s} | {'LEADER  min   now   max  span':^30s} | {'FOLLOWER  min   now   max  span':^31s}"]
            for j in RANGED:
                cols = []
                for a in ("leader", "follower"):
                    ok = "ok" if span[a][j] >= 0.95 * old_span[a][j] else ".."
                    cols.append(f"{mins[a][j]:6d}{pos[a][j]:6d}{maxs[a][j]:6d}{span[a][j]:6d} {ok}")
                lines.append(f"  {j:14s} | {cols[0]:>30s} | {cols[1]:>31s}")
            lines += ["", "ok = at least 95% of the previous calibration's range (a guide; the hard stops are what count).",
                      "Press ENTER when every joint has been to both stops.", msg]
            show(lines)
            k = key()
            if k is not None:
                short = [f"{a} {j}" for a in buses for j in RANGED if span[a][j] < 0.85 * old_span[a][j]]
                wrap = [f"{a} {j}" for a in buses for j in RANGED if mins[a][j] < 40 or maxs[a][j] > 4055]
                if wrap:
                    raise SystemExit(f"\nABORT: {', '.join(wrap)} came near the encoder wrap; the middle pose was too far off. Run again.")
                if short and k != "f":
                    msg = f"STILL SHORT (<85% of before): {', '.join(short)}. Keep moving, or type f + ENTER to finish anyway."
                    continue
                break
            time.sleep(0.05)

        # ---- 3. SAVE -------------------------------------------------------------------------------
        new: dict[str, dict[str, MotorCalibration]] = {}
        for a, b in buses.items():
            cal = {}
            for j in JOINTS:
                lo, hi = ROLL[a] if j == "wrist_roll" else (mins[a][j], maxs[a][j])
                cal[j] = MotorCalibration(id=b.motors[j].id, drive_mode=0, homing_offset=homings[a][j],
                                          range_min=int(lo), range_max=int(hi))
            b.write_calibration(cal)
            back = b.read_calibration()
            bad = [j for j in JOINTS if as_dict({j: back[j]})[j] != as_dict({j: cal[j]})[j]]
            if bad:
                raise SystemExit(f"\nABORT: {a} servos did not keep the new values for {bad}")
            new[a] = cal
        for a, p in FILES.items():
            p.write_text(json.dumps(as_dict(new[a]), indent=4))
        saved = True

        if HOME_FILE.exists():   # keep the same physical HOME (roll 0 is HOME by definition)
            h = json.loads(HOME_FILE.read_text())
            oldf, newf = old_files["follower"], as_dict(new["follower"])
            prev, out = list(h["home"]), []
            for j, n in zip(h["joints"], h["home"]):
                if j == "wrist_roll":   # HOME = follower turned ROLL_TURN deg = leader straight
                    out.append(round(norm(j, 2047 + ROLL_TURN * TPD, newf[j]), 3)); continue
                actual = to_raw(j, n, oldf[j]) + old_motor["follower"][j].homing_offset
                raw_new = (actual - newf[j]["homing_offset"]) % 4096
                out.append(round(norm(j, raw_new, newf[j]), 3))
            h["previous_home"] = prev
            h["home"] = out
            h["source"]["note"] = (f"Final calibration {stamp}: HOME converted to the new calibration, same physical pose. "
                                   f"wrist_roll ranges: follower {ROLL['follower']}, leader {ROLL['leader']} "
                                   f"(follower turned {ROLL_TURN} deg vs leader).")
            h["source"]["changed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            HOME_FILE.write_text(json.dumps(h, indent=1))

        report = [f"FINAL CALIBRATION {stamp}", f"{'joint':14s} {'L span':>7s} {'F span':>7s} {'F/L':>6s}  L min..max   F min..max"]
        for j in JOINTS:
            l, f = as_dict(new["leader"])[j], as_dict(new["follower"])[j]
            ls, fs = l["range_max"] - l["range_min"], f["range_max"] - f["range_min"]
            report.append(f"{j:14s} {ls:7d} {fs:7d} {fs / ls:6.3f}  {l['range_min']:4d}..{l['range_max']:4d}  {f['range_min']:4d}..{f['range_max']:4d}")
        if HOME_FILE.exists():
            report.append(f"HOME before: {prev}")
            report.append(f"HOME now:    {out}")
        report.append(f"backup: {backup}")
        (ROOT / "logs" / f"calib-final-{stamp}.txt").write_text("\n".join(report) + "\n")

        # ---- 4. VERIFY -----------------------------------------------------------------------------
        nl, nf = as_dict(new["leader"]), as_dict(new["follower"])
        while True:
            pos = read_all(buses)
            lines = ["=== STEP 4/4  VERIFY (saved) ===  both arms limp", "",
                     "Put both arms in the SAME pose (follower wrist turned 90 deg vs the leader). Diffs should be small.",
                     f"  {'joint':14s} {'LEADER':>8s} {'FOLLOWER':>9s} {'diff':>7s}    (Studio units: -100..100, gripper 0..100)"]
            for j in JOINTS:
                l, f = norm(j, pos["leader"][j], nl[j]), norm(j, pos["follower"][j], nf[j])
                flag = "  <-- check" if abs(f - l) > 5 else ""
                lines.append(f"  {j:14s} {l:+8.1f} {f:+9.1f} {f - l:+7.1f}{flag}")
            lines += ["", *report[:8], "", "Press ENTER to continue to the teleop test."]
            show(lines)
            if key() is not None:
                break
            time.sleep(0.1)
        return 0
    except BaseException as e:
        if not saved:
            for a, b in buses.items():
                b.disable_torque()
                b.write_calibration(old_motor[a])
            print(f"\n{e if isinstance(e, SystemExit) else 'Stopped.'}  Old servo values written back; nothing saved.")
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            return 1
        raise
    finally:
        for b in buses.values():
            b.disconnect(disable_torque=True)


if __name__ == "__main__":
    sys.exit(main())
