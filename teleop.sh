#!/usr/bin/env bash
B=~/miniforge3/envs/intel_dev_env/bin
LOG=~/checkurai/logs/teleop-$(date +%H%M%S).log
clear; echo "VERIFY first (both arms limp): same pose -> numbers should match. Reversed joints: $(cat ~/checkurai/tools/invert.json)"
echo "(errors are saved to $LOG)"
read -rp "Press ENTER... "
$B/python ~/checkurai/tools/verify_arms.py 2> >(tee -a "$LOG" >&2) || echo "VERIFY ERROR (exit $?) - see $LOG"
echo; echo "TELEOP: leader in same pose as follower; hands away from follower. Ctrl+C stops (follower goes LIMP, hold it)."
read -rp "Press ENTER to start teleop... "
$B/python ~/checkurai/tools/teleop_inv.py 2> >(tee -a "$LOG" >&2) || echo "TELEOP ERROR (exit $?) - see $LOG"
read -rp $'\nENTER to close. (Run again: bash ~/checkurai/teleop.sh)'
