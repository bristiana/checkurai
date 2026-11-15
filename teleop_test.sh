#!/usr/bin/env bash
B=~/miniforge3/envs/intel_dev_env/bin
LOG=~/checkurai/logs/teleop-$(date +%H%M%S).log
clear
echo "TELEOP (final calibration 14:21)."
echo " - The follower copies the leader 1:1 (wrist: leader straight = follower HOME roll, 10 deg left)."
echo " - The base (shoulder_pan) now uses the same range on both arms."
echo " - It ramps slowly (4 s) to the leader pose first. Ctrl+C stops (the follower goes LIMP, so hold it)."
echo; read -rp "Leader in about the follower's pose, hands away from the follower. ENTER to start... "
$B/python ~/checkurai/tools/teleop_inv.py 2> >(tee -a "$LOG" >&2) || echo "TELEOP ERROR - see $LOG"
read -rp $'\nENTER to close. (Run again: bash ~/checkurai/teleop_test.sh)'
