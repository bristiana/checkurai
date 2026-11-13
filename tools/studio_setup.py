#!/usr/bin/env python3
"""Register the CheckurAI station in Physical AI Studio through its REST API.

Idempotent: reuses the project, cameras, robots and environment if they already exist (matched by name).
Robots are added by USB serial with our LeRobot calibration, so Studio skips its calibration wizard and
never writes calibration to the servos.

    python3 tools/studio_setup.py            # create / reuse everything, print the ids
    python3 tools/studio_setup.py --dry-run  # show what would be sent
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

API = "http://127.0.0.1:7860/api"
PROJECT = "CheckurAI"
CAL_DIR = Path.home() / ".cache/huggingface/lerobot/calibration"
ROBOTS = {
    "follower": ("SO101_Follower", "5AE7045345", CAL_DIR / "robots/so_follower/checkurai_follower.json"),
    "leader": ("SO101_Leader", "5B90148356", CAL_DIR / "teleoperators/so_leader/checkurai_leader.json"),
}
# name -> (driver, how to recognise it in /hardware/cameras, payload)
CAMERAS = {
    "top": ("realsense", lambda c: c["driver"] == "realsense", {"width": 640, "height": 480, "fps": 30, "output_type": "color"}),
    "wrist": ("usb_camera", lambda c: c["driver"] == "usb_camera" and "USB2.0_CAM1" in c["name"], {"width": 640, "height": 480, "fps": 30}),
    "inspection": ("usb_camera", lambda c: c["driver"] == "usb_camera" and c["name"].startswith("Webcam"), {"width": 640, "height": 480, "fps": 30}),
}
ENVIRONMENTS = {"pick": ["top", "wrist"]}

DRY = False


def call(method: str, path: str, body: dict | None = None):
    if DRY and method != "GET":
        print(f"[dry-run] {method} {path} {json.dumps(body)[:300]}")
        return {"id": str(uuid.uuid4()), **(body or {})}
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} -> {e.code}: {e.read().decode()[:2000]}")


def by_name(items: list[dict], name: str) -> dict | None:
    return next((i for i in items if i.get("name") == name), None)


def main() -> None:
    global DRY
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    DRY = ap.parse_args().dry_run

    project = by_name(call("GET", "/projects"), PROJECT) or call("POST", "/projects", {"id": str(uuid.uuid4()), "name": PROJECT})
    pid = project["id"]
    print(f"project {PROJECT}: {pid}")

    hw = call("GET", "/hardware/cameras")
    existing = call("GET", f"/projects/{pid}/cameras")
    cam_ids = {}
    for name, (driver, match, payload) in CAMERAS.items():
        cam = by_name(existing, name)
        if cam is None:
            found = [c for c in hw if match(c)]
            if len(found) != 1:
                sys.exit(f"camera {name}: expected 1 match, found {len(found)} in {[c['name'] for c in hw]}")
            c = found[0]
            cam = call("POST", f"/projects/{pid}/cameras", {
                "id": str(uuid.uuid4()), "driver": driver, "name": name,
                "fingerprint": c["fingerprint"], "hardware_name": c["name"], "payload": payload})
        cam_ids[name] = cam["id"]
        print(f"camera {name}: {cam['id']}")

    existing = call("GET", f"/projects/{pid}/robots")
    robot_ids = {}
    for name, (rtype, serial, cal_path) in ROBOTS.items():
        robot = by_name(existing, name)
        if robot is None:
            calibration = json.loads(cal_path.read_text())
            robot = call("POST", f"/projects/{pid}/robots", {
                "id": str(uuid.uuid4()), "name": name, "type": rtype,
                "payload": {"connection_string": "", "serial_number": serial, "unit": "normalized",
                            "calibration": calibration}})
        robot_ids[name] = robot["id"]
        print(f"robot {name}: {robot['id']}")

    existing = call("GET", f"/projects/{pid}/environments")
    for name, cams in ENVIRONMENTS.items():
        env = by_name(existing, name)
        if env is None:
            env = call("POST", f"/projects/{pid}/environments", {
                "id": str(uuid.uuid4()), "name": name,
                "robots": [{"robot_id": robot_ids["follower"],
                            "tele_operator": {"type": "robot", "robot_id": robot_ids["leader"]}}],
                "cameras": [{"camera_id": cam_ids[c]} for c in cams]})
        print(f"environment {name}: {env['id']}")


if __name__ == "__main__":
    main()
