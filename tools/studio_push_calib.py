#!/usr/bin/env python3
"""Copy our LeRobot calibration files into Physical AI Studio's robots (follower + leader).

Studio refuses to connect when its stored calibration differs from the servo registers, so run this
after every calibration (tools/calib_pair.py writes the servos and the files).

    python3 tools/studio_push_calib.py
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

CELL = json.loads((Path(__file__).resolve().parent.parent / "config/cell.json").read_text())
API = f"{CELL['url']}/api/projects/{CELL['project']}/robots"
CAL = Path.home() / ".cache/huggingface/lerobot/calibration"
FILES = {"follower": CAL / "robots/so_follower/checkurai_follower.json",
         "leader": CAL / "teleoperators/so_leader/checkurai_leader.json"}

ok = True
for robot in json.load(urllib.request.urlopen(API)):
    path = FILES.get(robot["name"])
    if path is None:
        continue
    cal = json.loads(path.read_text())
    body = {k: robot[k] for k in ("id", "name", "type", "payload")}
    body["payload"]["calibration"] = cal
    req = urllib.request.Request(f"{API}/{robot['id']}", data=json.dumps(body).encode(), method="PUT",
                                 headers={"Content-Type": "application/json"})
    try:
        out = json.load(urllib.request.urlopen(req))["payload"]["calibration"]
    except urllib.error.HTTPError as e:
        print(robot["name"], "ERROR", e.code, e.read()[:400]); ok = False; continue
    same = out == cal
    ok &= same
    print(f"{robot['name']:9s} {'updated' if same else 'MISMATCH after PUT'}: "
          + "  ".join(f"{j}={c['range_min']}..{c['range_max']}" for j, c in out.items()))
sys.exit(0 if ok else 1)
