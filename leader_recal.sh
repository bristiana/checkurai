#!/usr/bin/env bash
B=~/miniforge3/envs/intel_dev_env/bin
L=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00
CAL=~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader
clear
echo "=== STEP 1/3: calibrate the LEADER (blue). Nothing moves by itself; the red arm stays off. ==="
echo " a) Put the BLUE arm in the MIDDLE pose (every joint in the middle of its range) and press ENTER."
echo " b) Then slowly move EVERY joint of the blue arm through its FULL range (wrist too, gripper"
echo "    fully open and closed) and press ENTER."
echo
[ -f $CAL/checkurai_leader.json ] && mv $CAL/checkurai_leader.json $CAL/checkurai_leader.json.bak-$(date +%H%M%S)
$B/lerobot-calibrate --teleop.type=so101_leader --teleop.port=$L --teleop.id=checkurai_leader || { read -rp "Leader calibration failed (copy the error to Claude). ENTER to close"; exit 1; }
echo; echo "=== STEP 2/3: DIRECTION TEST (both arms limp). One joint at a time, same way on BOTH arms. ==="
echo "    Every joint should say SAME. ENTER when done."
read -rp "Press ENTER to start the test... "
$B/python ~/checkurai/tools/direction_test.py || echo "TEST ERROR (exit $?)"
echo; read -rp "=== STEP 3/3: ENTER to go to teleop (verify first), or close this window if a joint said OPPOSITE. "
exec bash ~/checkurai/teleop.sh
