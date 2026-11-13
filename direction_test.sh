#!/usr/bin/env bash
clear; echo "DIRECTION TEST - both arms stay LIMP (no motors)."
echo "Hold the BLUE arm in one hand and the RED arm in the other."
echo "Move ONE joint at a time, the SAME way on both (e.g. both wrists bend up), back and forth a few times."
read -rp "Press ENTER to start... "
~/miniforge3/envs/intel_dev_env/bin/python ~/checkurai/tools/direction_test.py || echo "ERROR (exit $?)"
read -rp $'\nENTER to close.'
