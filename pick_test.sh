#!/bin/bash
# Live test of one policy through the cell program (30 Hz, OpenVINO on the GPU, stops when the arm is
# back at HOME and still). Runs once per Enter. Close Studio's inference / record page first.
#     bash pick_test.sh [model] [extra orchestrator options]          # pick: brick on the plate
#     STAGE=place_good bash pick_test.sh [model] [options]            # place: brick on the inspection spot
#     e.g. bash pick_test.sh act-pick-v3 --request-threshold 0.5      # compare with the old re-plan rate
cd "$(dirname "$0")" || exit 1
export PYTHONUNBUFFERED=1
PY=$HOME/miniforge3/envs/intel_dev_env/bin/python
if [[ -z $STAGE ]]; then  # a place model run as the pick stage tries to pick from the plate: take the stage from the name
  case "${1:-}" in
    *place-good*) STAGE=place_good ;;
    *place-bad*) STAGE=place_bad ;;
    *place*) echo "say which bowl $1 should use: STAGE=place_good bash pick_test.sh $1  (or STAGE=place_bad)"; exit 1 ;;
    *) STAGE=pick ;;
  esac
  echo "stage: $STAGE (from the model name)"
fi
MODEL=${1:-act-${STAGE//_/-}-v3}
shift
EXTRA=("$@")
[[ $STAGE == pick ]] && WHERE="ONE brick on the plate" || WHERE="ONE brick on the inspection spot"
TUNE=$(~/physical-ai-studio/application/backend/.venv/bin/python tools/policy_tune.py "$MODEL" --show 2>/dev/null | sed 's/.*: //')
pgrep -f "anomaly_train.py" >/dev/null && echo "note: an anomaly job is also using the GPU (the policy still runs, a bit slower)"
read -r -p "Press Enter to start (the arm goes HOME first)... " _
while true; do
  echo
  echo "=== $STAGE test with $MODEL ${EXTRA[*]} (gripper: $TUNE) ==="
  echo "Put $WHERE. The arm goes HOME by itself first (hand near the 12 V switch)."
  echo "When it asks, type go + Enter. While it runs: s + Enter stops and holds the arm."
  if ! $PY tools/go_home.py; then
    echo "could not bring the arm HOME (is a Studio page still holding it?)"
    read -r -p "Enter = try again, q + Enter = quit: " a; [[ $a == q ]] && break; continue
  fi
  # the top camera drifts: re-fit the model's camera correction to where it points now (skips if unchanged)
  echo "checking the top camera against $MODEL's demos (a few s; ~1-2 min for a VLA when it rebuilds)..."
  ~/physical-ai-studio/application/backend/.venv/bin/python tools/policy_tune.py "$MODEL" --top-warp "${WARP:-auto}" 2>&1 \
    | grep -E "live camera vs|unchanged|tuned:|WARNING|not applied|only markers|no ArUco" \
    || echo "top camera check failed - running with the previous correction"
  $PY tools/orchestrator.py "--${STAGE//_/-}-model" "$MODEL" "${EXTRA[@]}" run --only "$STAGE" --no-inspection 2>&1 \
    | tee -a logs/pick_test.log
  echo
  read -r -p "Enter = run again, q + Enter = quit: " a
  [[ $a == q ]] && break
done
