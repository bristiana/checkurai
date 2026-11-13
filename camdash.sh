#!/usr/bin/env bash
# Camera dashboard control: bash camdash.sh [start|stop|restart|status]
cd "$(dirname "$0")"
PY=~/miniforge3/envs/intel_dev_env/bin/python
running() { pgrep -f "python tools/camdash.py" >/dev/null; }
start() {
  if running; then echo "already running"; else
    mkdir -p logs; (setsid nohup "$PY" tools/camdash.py > logs/camdash.log 2>&1 < /dev/null &); sleep 2
    running && echo "started on http://localhost:8080" || { echo "failed:"; tail -5 logs/camdash.log; }
  fi
}
stop() { pkill -f "python tools/camdash.py" && echo "stopped" || echo "not running"; }
case "${1:-status}" in
  start) start ;; stop) stop ;; restart) stop; sleep 1; start ;;
  status) running && { echo "running"; curl -s -m 3 http://127.0.0.1:8080/stats; echo; } || echo "not running" ;;
  *) echo "usage: $0 [start|stop|restart|status]" ;;
esac
