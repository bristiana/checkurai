#!/usr/bin/env bash
cd ~/checkurai/tools
~/miniforge3/envs/intel_dev_env/bin/python roll_setup.py && python3 studio_push_calib.py && echo && echo "Wrist set. Teleop test: bash ~/checkurai/teleop_test.sh"
read -rp $'\nENTER to close.'
