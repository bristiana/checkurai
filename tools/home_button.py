"""Web page with a HOME button for the follower arm.

Opens the follower's serial port only while a command runs, so it never fights
Studio for the arm. Close Studio's record page (its runtime session) first.

    python tools/home_button.py            # serves http://127.0.0.1:8765

Motion: release the gripper, fold shoulder/elbow/wrist to HOME, then turn
pan/roll to HOME, then close the gripper. Joint units match Studio
(body -100..100, gripper 0..100). STOP holds the arm where it is with torque
on; RELEASE turns torque off (support the arm first).
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00"
FOLLOWER_ID = "checkurai_follower"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FOLD = ["shoulder_lift", "elbow_flex", "wrist_flex"]
TURN = ["shoulder_pan", "wrist_roll"]
SPEED = 30.0  # units per second for the largest joint move in a phase
RATE = 30.0
GRIP_RELEASE = 30.0

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Arm HOME</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px;display:grid;gap:14px;max-width:520px}
button{font-size:28px;padding:22px;border:0;border-radius:12px;font-weight:700;cursor:pointer}
#home{background:#1f9a63;color:#fff}#stop{background:#d6452f;color:#fff}#rel{background:#444;color:#eee;font-size:18px;padding:12px}
#read{background:#2b4b7a;color:#fff;font-size:18px;padding:12px}
pre{background:#000;padding:12px;border-radius:8px;white-space:pre-wrap;font-size:15px;min-height:150px}
</style>
<h2>Follower arm</h2>
<button id=home onclick="go('home')">GO HOME</button>
<button id=stop onclick="go('stop')">STOP (hold here)</button>
<button id=read onclick="go('read')">Read pose</button>
<button id=rel onclick="if(confirm('Torque OFF: the arm goes limp. Support it first. OK?'))go('release')">Release torque</button>
<pre id=s>...</pre>
<script>
async function go(c){await fetch('/'+c,{method:'POST'});}
async function poll(){try{const r=await fetch('/status');document.getElementById('s').textContent=await r.text();}catch(e){}setTimeout(poll,400);}
poll();
</script>"""


def port_users(port: str) -> list[int]:
    """PIDs of other processes that have the serial device open."""
    import os

    dev = os.path.realpath(port)
    me = os.getpid()
    users = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.path.realpath(f"/proc/{pid}/fd/{fd}") == dev:
                    users.append(int(pid))
                    break
        except OSError:
            continue
    return users


class Arm:
    def __init__(self, home: dict[str, float]):
        self.home = home
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.robot = None  # connected follower while holding
        self.status = "idle. Port closed; Studio can use the arm."

    def log(self, msg: str) -> None:
        self.status = time.strftime("%H:%M:%S ") + msg
        print(self.status, flush=True)

    def _open_bus(self):
        """Open the follower bus, refusing if any other process has the port open (e.g. Studio)."""
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        users = port_users(FOLLOWER_PORT)
        if users:
            raise RuntimeError(
                f"the arm is in use by another program (pid {', '.join(map(str, users))}). "
                "In Studio press Stop, then close the record page (or switch to another page), and try again."
            )
        robot = SOFollower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, use_degrees=False))
        try:
            robot.bus.connect()
        except Exception as exc:  # unplugged, no power, or motors not answering
            try:
                robot.bus.port_handler.closePort()
            except Exception:
                pass
            raise RuntimeError(f"cannot talk to the follower ({exc}). Is the 12 V power on?") from exc
        return robot

    def _connect(self):
        if self.robot is not None:
            return self.robot
        robot = self._open_bus()
        # Make the goal equal to where the arm is, so enabling torque does not jump.
        robot.bus.disable_torque()
        raw = robot.bus.sync_read("Present_Position", normalize=False)
        robot.bus.sync_write("Goal_Position", raw, normalize=False)
        robot.configure()
        self.robot = robot
        return robot

    def _pose(self, robot) -> dict[str, float]:
        pos = robot.bus.sync_read("Present_Position")
        return {j: float(pos[j]) for j in JOINTS}

    def _fmt(self, pose: dict[str, float]) -> str:
        lines = [f"{'joint':14s} {'now':>7s} {'HOME':>7s} {'diff':>7s}"]
        for j in JOINTS:
            lines.append(f"{j:14s} {pose[j]:7.1f} {self.home[j]:7.1f} {pose[j] - self.home[j]:7.1f}")
        return "\n".join(lines)

    def _move(self, robot, target: dict[str, float], names: list[str]) -> bool:
        start = self._pose(robot)
        delta = max(abs(target[j] - start[j]) for j in names)
        if delta < 0.5:
            return True
        duration = max(1.2, delta / SPEED)
        steps = int(duration * RATE)
        for i in range(1, steps + 1):
            if self.stop_flag.is_set():
                return False
            s = 0.5 - 0.5 * math.cos(math.pi * i / steps)  # smooth start and stop
            action = {f"{j}.pos": start[j] + (target[j] - start[j]) * s for j in names}
            robot.send_action(action)
            time.sleep(1.0 / RATE)
        return True

    def _hold_here(self, robot) -> None:
        pose = self._pose(robot)
        robot.send_action({f"{j}.pos": pose[j] for j in JOINTS})

    def _disconnect(self, torque_off: bool) -> None:
        if self.robot is not None:
            try:
                self.robot.bus.disconnect(disable_torque=torque_off)
            finally:
                self.robot = None

    def go_home(self) -> None:
        if not self.lock.acquire(blocking=False):
            self.log("busy")
            return
        self.stop_flag.clear()
        try:
            robot = self._connect()
            pose = self._pose(robot)
            self.log("moving to HOME\n" + self._fmt(pose))
            ok = True
            if pose["gripper"] < GRIP_RELEASE:
                ok = self._move(robot, {**pose, "gripper": GRIP_RELEASE}, ["gripper"])
            ok = ok and self._move(robot, self.home, FOLD)
            ok = ok and self._move(robot, self.home, TURN)
            ok = ok and self._move(robot, self.home, ["gripper"])
            if not ok:
                self._hold_here(robot)
                self.log("STOPPED: holding here with torque ON. Press GO HOME or Release torque.\n" + self._fmt(self._pose(robot)))
                return
            time.sleep(0.5)
            pose = self._pose(robot)
            far = [j for j in JOINTS if abs(pose[j] - self.home[j]) > 8]
            self._disconnect(torque_off=True)
            msg = "at HOME. Torque off, port closed; Studio can use the arm."
            if far:
                msg = "near HOME but off on " + ", ".join(far) + " (blocked?). Torque off, port closed."
            self.log(msg + "\n" + self._fmt(pose))
        except Exception as exc:
            self.log(f"ERROR: {exc}")
            if self.robot is not None:
                try:
                    self._hold_here(self.robot)
                except Exception:
                    pass
        finally:
            self.lock.release()

    def read(self) -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            if self.robot is not None:
                self.log("holding\n" + self._fmt(self._pose(self.robot)))
                return
            robot = self._open_bus()
            try:
                pose = self._pose(robot)
            finally:
                robot.bus.disconnect(disable_torque=False)  # read only: leave torque as it was
            self.log("read (torque unchanged, port closed)\n" + self._fmt(pose))
        except Exception as exc:
            self.log(f"ERROR: {exc}")
        finally:
            self.lock.release()

    def release(self) -> None:
        self.stop_flag.set()
        with self.lock:
            try:
                if self.robot is None:
                    robot = self._open_bus()
                    robot.bus.disconnect(disable_torque=True)
                else:
                    self._disconnect(torque_off=True)
                self.log("torque OFF, port closed.")
            except Exception as exc:
                self.log(f"ERROR: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    cfg = json.loads((ROOT / "config/home_pose.json").read_text())
    arm = Arm(dict(zip(cfg["joints"], cfg["home"])))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, body: str, ctype: str = "text/plain; charset=utf-8") -> None:
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/status":
                self._send(arm.status)
            else:
                self._send(PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            if self.path == "/home":
                threading.Thread(target=arm.go_home, daemon=True).start()
            elif self.path == "/stop":
                arm.stop_flag.set()
            elif self.path == "/read":
                threading.Thread(target=arm.read, daemon=True).start()
            elif self.path == "/release":
                threading.Thread(target=arm.release, daemon=True).start()
            self._send("ok")

    print(f"HOME button on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
