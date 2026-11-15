#!/usr/bin/env bash
# Final calibration of both arms (tools/calib_pair.py), then the teleop test (tools/teleop_inv.py).
B=~/miniforge3/envs/intel_dev_env/bin
LOG=~/checkurai/logs/calib-final-$(date +%H%M%S).log
clear
echo "FINAL CALIBRATION of both arms. Both go LIMP now: hold the follower."
echo "(errors are saved to $LOG)"
read -rp "Press ENTER to start... "
clear
if $B/python -u ~/checkurai/tools/calib_pair.py 2> >(tee -a "$LOG" >&2); then
  echo; echo "TELEOP TEST: the follower mirrors the leader, with its wrist turned 90 deg. Ctrl+C stops (the follower goes LIMP, so hold it)."
  read -rp "Leader in about the follower's pose, hands away from the follower. ENTER to start teleop... "
  $B/python ~/checkurai/tools/teleop_inv.py 2> >(tee -a "$LOG" >&2) || echo "TELEOP ERROR - see $LOG"
else
  echo "Calibration not saved (see above / $LOG)."
fi
read -rp $'\nENTER to close. (Run again: bash ~/checkurai/final_calib.sh)'
