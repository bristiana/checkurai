#!/bin/bash
# Guided inspection photo session for the anomaly model: empty spot, 10 good bricks, 4 defect kinds.
# Each step is one capture_inspection.py run; the status page (http://127.0.0.1:8766) shows what to do,
# the count, and a "Next" button that ends a step early. Step start/end times go to
# logs/capture_session.log (HHMMSS, same clock as the photo names) to fill GOOD_BRICKS / DEFECT_TYPES
# in tools/anomaly_train.py.
#
#     bash tools/capture_session.sh                # all steps
#     bash tools/capture_session.sh defect         # only the steps whose name starts with "defect"
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$HOME/miniforge3/envs/intel_dev_env/bin/python}
DEV=${DEV:-/dev/video32}
PORT=${PORT:-8766}
LOG=logs/capture_session.log
ONLY=$1
mkdir -p logs

step() {  # name label max note [extra args]
  local name=$1 label=$2 max=$3 note=$4; shift 4
  [[ -n $ONLY && $name != $ONLY* ]] && return
  echo "$(date +%H%M%S) start $name $label" | tee -a $LOG
  $PY tools/capture_inspection.py --device "$DEV" --port "$PORT" --label "$label" --max "$max" --note "$note" "$@"
  echo "$(date +%H%M%S) end $name $label" | tee -a $LOG
  sleep 2
}

while fuser "$DEV" >/dev/null 2>&1; do  # Studio (a page with a live camera view) still holds the camera
  [[ -z $warned ]] && echo "$DEV is busy (close the Studio tab that shows the cameras); waiting..." && warned=1
  sleep 2
done

if [[ $ONLY == markers ]]; then  # extra marker-only defect round (bash tools/capture_session.sh markers)
  M="Move and turn the brick each time; use different bricks and put the marks on different faces (top, front, sides, back)"
  ONLY=
  step marker_line2 defect 20 "DEFECT: marker LINES. $M"
  step marker_bigdot defect 20 "DEFECT: BIG marker DOTS. $M"
  step marker_smalldot defect 20 "DEFECT: SMALL marker DOTS. $M"
  echo "$(date +%H%M%S) session done" | tee -a $LOG
  exit 0
fi

step empty empty 20 "EMPTY spot: move your hand over the spot and take it away. Repeat until 20." --same -1
for i in $(seq 1 10); do
  step "good$i" good 10 "GOOD brick $i of 10 (a new brick): 6 studs up, moved and turned each time; then 2 upside down, 2 on its side"
done
step defect_tape defect 10 "DEFECT: brick with TAPE on it. Move and turn it each time; use 2-3 different bricks"
step defect_marker_line defect 10 "DEFECT: brick with a MARKER LINE. Move and turn it each time; use 2-3 different bricks"
step defect_marker_dot defect 10 "DEFECT: brick with a MARKER DOT. Move and turn it each time; use 2-3 different bricks"
step defect_other defect 10 "DEFECT: any other defect (broken, sticker, paper, dirt). Press Next if you have none"
echo "$(date +%H%M%S) session done" | tee -a $LOG
