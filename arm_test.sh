#!/usr/bin/env bash
# CheckurAI v3: identify joints -> calibrate (follower always redone, leader optional) -> verify -> teleop
B=~/miniforge3/envs/intel_dev_env/bin
L=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00
F=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00
CAL=~/.cache/huggingface/lerobot/calibration
FJ=$CAL/robots/so_follower/checkurai_follower.json; LJ=$CAL/teleoperators/so_leader/checkurai_leader.json
clear
echo "STEP 1/4: IDENTIFY. Move ONE joint at a time on each arm and check the right row says MOVING."
read -rp "Press ENTER to open the identify view... "
$B/python ~/checkurai/tools/identify.py
read -rp $'\nWere ALL rows correct on BOTH arms? Type y + ENTER to continue (anything else = stop): ' ok
[ "$ok" = "y" ] || { read -rp "Stopped. Tell Claude which rows were wrong. ENTER to close"; exit 1; }
cat <<T

STEP 2/4: CALIBRATION. SAME middle pose for BOTH arms:
  base straight forward | upper arm straight up | forearm horizontal forward (elbow ~90)
  wrist straight | gripper jaws level | gripper half open
Range step: move EVERY joint to BOTH end stops (incl. base L/R, shoulder up/down,
GRIPPER fully open + closed) until MIN and MAX are far apart on every row. Then ENTER.
T
rm -f "$FJ"
read -rp $'\nPress ENTER to calibrate the FOLLOWER... '
$B/lerobot-calibrate --robot.type=so101_follower --robot.port=$F --robot.id=checkurai_follower || { read -rp "Follower calibration failed. ENTER to close"; exit 1; }
if [ -f "$LJ" ]; then
  read -rp $'\nRedo the LEADER calibration too? (y = redo, ENTER = keep) ' a; [ "$a" = "y" ] && rm -f "$LJ"
fi
if [ ! -f "$LJ" ]; then
  echo "LEADER: SAME middle pose, then UNPLUG its 5 V power for 5 s and plug back (don't move it)."
  read -rp "Press ENTER when done... "
  $B/lerobot-calibrate --teleop.type=so101_leader --teleop.port=$L --teleop.id=checkurai_leader || { read -rp "Leader calibration failed. ENTER to close"; exit 1; }
fi
read -rp $'\nSTEP 3/4: VERIFY (both arms limp; same pose -> numbers match). Press ENTER... '
$B/python ~/checkurai/tools/verify_arms.py
echo; echo "STEP 4/4: TELEOP. Only if VERIFY matched. Leader in same pose as follower. Ctrl+C stops (follower goes LIMP, hold it)."
read -rp "Press ENTER to start teleop... "
$B/lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=$F --robot.id=checkurai_follower --robot.max_relative_target=10 \
  --teleop.type=so101_leader --teleop.port=$L --teleop.id=checkurai_leader --display_data=false
read -rp $'\nTeleop stopped. ENTER to close. (Run again: bash ~/checkurai/arm_test.sh)'
