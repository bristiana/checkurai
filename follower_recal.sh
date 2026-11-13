#!/usr/bin/env bash
B=~/miniforge3/envs/intel_dev_env/bin
F=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00
CAL=~/.cache/huggingface/lerobot/calibration/robots/so_follower
clear
echo "=== Base rotation (shoulder_pan) is off by ~27 deg: the RED arm's base was turned only 172 deg during"
echo "    calibration, the BLUE one 224 deg. Fix = recalibrate the RED arm (2 min, it stays limp)."
echo
read -rp "Type c + ENTER to recalibrate the RED arm (recommended), or just ENTER to skip to teleop: " ans
if [ "$ans" = "c" ]; then
  echo
  echo " a) RED arm in the MIDDLE pose (every joint mid-range), press ENTER."
  echo " b) Move EVERY joint of the RED arm to BOTH of its hard stops, slowly. Especially the BASE:"
  echo "    turn it fully left AND fully right (free the cables if they block it). Gripper fully open/closed. ENTER."
  echo
  [ -f $CAL/checkurai_follower.json ] && mv $CAL/checkurai_follower.json $CAL/checkurai_follower.json.bak-$(date +%H%M%S)
  $B/lerobot-calibrate --robot.type=so101_follower --robot.port=$F --robot.id=checkurai_follower || { read -rp "Calibration failed (copy the error to Claude). ENTER to close"; exit 1; }
fi
exec bash ~/checkurai/teleop.sh
