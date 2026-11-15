#!/usr/bin/env python3
"""Auto-capture inspection photos for the anomaly model (no keypress needed).

Put a brick on the spot and take your hand away. When the picture has changed (a hand came in)
and then stays still for about 0.8 s, one full-resolution photo is saved. Turn the brick, take
your hand away, and the next one is saved. If nothing really changed, no duplicate is saved.

    python3 tools/capture_inspection.py --label good
    python3 tools/capture_inspection.py --label defect_tape --max 20
    python3 tools/capture_inspection.py --label good --debug      # print the motion numbers

Photos go to data/inspection/<label>/<timestamp>.jpg (data/ is gitignored).
Live status page: http://127.0.0.1:8765  (from another machine: ssh -L 8765:localhost:8765 <laptop>)
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SMALL_W = 320  # analysis width (pixels); full-resolution frames are only used for saving


def find_camera_nodes(name: str) -> list[str]:
    """All /dev/videoN nodes whose V4L2 name contains `name` (lowest number first)."""
    base = Path("/sys/class/video4linux")
    nodes = []
    for p in sorted(base.glob("video*"), key=lambda p: int(p.name[5:])):
        try:
            if name.lower() in (p / "name").read_text().strip().lower():
                nodes.append(f"/dev/{p.name}")
        except OSError:
            pass
    return nodes


def open_camera(nodes: list[str], width: int, height: int, fps: int) -> tuple[cv2.VideoCapture, str]:
    for dev in nodes:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = False, None
        for _ in range(10):  # let auto exposure settle a little
            ok, frame = cap.read()
        if ok and frame is not None and frame.ndim == 3:
            return cap, dev
        cap.release()
    sys.exit(f"could not read frames from any of {nodes}. Is the camera busy? "
             "Close the camera preview in Studio and check with: fuser /dev/videoN")


def small_gray(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    g = cv2.cvtColor(cv2.resize(frame, (SMALL_W, round(SMALL_W * h / w)), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)
    return g - g.mean()  # ignore small global brightness shifts (auto exposure)


def changed_fraction(a: np.ndarray, b: np.ndarray, pix_t: float) -> float:
    return float((np.abs(a - b) > pix_t).mean())


def jpeg(img: np.ndarray, width: int | None = None, quality: int = 80) -> bytes:
    if width is not None and img.shape[1] > width:
        img = cv2.resize(img, (width, round(width * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


class Shared:
    def __init__(self, label: str, folder: Path):
        self.lock = threading.Lock()
        self.label, self.folder = label, folder
        self.state = "starting"
        self.count_run = 0
        self.count_total = len(list(folder.glob("*.jpg")))
        self.saved: list[tuple[Path, np.ndarray]] = []  # (photo, its small grey image) for this run
        self.last_jpg: bytes | None = None
        self.live_jpg: bytes | None = None
        self.resolution = ""
        self.device = ""
        self.note = ""  # instruction line shown on the page (--note)
        self.max = 0
        self.next_requested = threading.Event()  # "Next" button: end this run early

    def status(self) -> dict:
        with self.lock:
            return {"label": self.label, "note": self.note, "max": self.max,
                    "state": self.state, "saved_this_run": self.count_run,
                    "total_in_folder": self.count_total, "last": self.saved[-1][0].name if self.saved else None,
                    "resolution": self.resolution, "device": self.device, "folder": str(self.folder)}

    def undo(self) -> str | None:
        with self.lock:
            if not self.saved:
                return None
            p, _ = self.saved.pop()
            p.unlink(missing_ok=True)
            self.count_run -= 1
            self.count_total -= 1
            self.last_jpg = jpeg(cv2.imread(str(self.saved[-1][0])), 640) if self.saved else None
            return p.name


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Inspection capture</title>
<style>
body{font:15px system-ui,sans-serif;margin:16px;background:#111;color:#eee}
h1{font-size:20px;margin:0 0 8px} .row{display:flex;gap:16px;flex-wrap:wrap}
figure{margin:0;flex:1 1 420px} img{width:100%;border-radius:6px;background:#222;min-height:120px}
#state{font-weight:700;padding:2px 10px;border-radius:4px}
.waiting{background:#345} .motion{background:#a60} .saved{background:#2a6} .starting{background:#444}
button{font:inherit;padding:6px 14px;margin-left:12px}
#note{font-size:22px;font-weight:700;background:#235;padding:10px 14px;border-radius:6px;margin:0 0 10px}
#note:empty{display:none}
</style>
<h1>Inspection capture: <span id=label></span></h1>
<p id=note></p>
<p><span id=state></span> &nbsp; saved this run: <b id=run></b> &nbsp; in folder: <b id=total></b>
&nbsp; <span id=res></span> <button id=undo>Delete last photo</button> <button id=next>Next (skip the rest of this step)</button></p>
<div class=row>
<figure><figcaption>Last saved photo: <span id=last></span></figcaption><img id=lastimg alt="last saved photo"></figure>
<figure><figcaption>Live</figcaption><img id=live alt="live view"></figure>
</div>
<script>
let shown=null;
async function tick(){
  try{
    const s=await (await fetch('/status.json',{cache:'no-store'})).json();
    label.textContent=s.label; state.textContent=s.state; state.className=s.state;
    note.textContent=s.note||''; run.textContent=s.saved_this_run+(s.max?' / '+s.max:''); total.textContent=s.total_in_folder;
    res.textContent=s.resolution+' '+s.device; last.textContent=s.last||'none yet';
    if(s.last!==shown){shown=s.last; lastimg.src=s.last?'/last.jpg?'+encodeURIComponent(s.last):'';}
    live.src='/live.jpg?'+Date.now();
  }catch(e){state.textContent='not running'; state.className='starting';}
}
undo.onclick=async()=>{await fetch('/undo',{method:'POST'}); tick();};
next.onclick=async()=>{await fetch('/next',{method:'POST'}); tick();};
setInterval(tick,1000); tick();
</script>"""


def make_handler(shared: Shared):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/status.json":
                self._send(200, json.dumps(shared.status()).encode(), "application/json")
            elif path in ("/last.jpg", "/live.jpg"):
                with shared.lock:
                    body = shared.last_jpg if path == "/last.jpg" else shared.live_jpg
                if body:
                    self._send(200, body, "image/jpeg")
                else:
                    self._send(404, b"no image yet", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/undo":
                name = shared.undo()
                print(f"deleted {name}" if name else "nothing to delete", flush=True)
                self._send(200, json.dumps({"deleted": name}).encode(), "application/json")
            elif self.path == "/next":
                shared.next_requested.set()
                print("Next pressed: ending this run", flush=True)
                self._send(200, b"{}", "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args):  # keep the terminal quiet
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, help="class folder name, e.g. good, defect_tape")
    ap.add_argument("--max", type=int, default=0, help="stop after N photos in this run (0 = no limit)")
    ap.add_argument("--camera-name", default="Webcam Audio", help="part of the V4L2 camera name")
    ap.add_argument("--device", help="force a /dev/videoN instead of searching by name")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "inspection")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--still-s", type=float, default=0.8, help="seconds the picture must stay still before saving")
    ap.add_argument("--pix-t", type=float, default=18, help="grey-level change that counts as a changed pixel")
    ap.add_argument("--motion", type=float, default=0.01, help="changed fraction vs the last saved photo that arms a save (a hand came in)")
    ap.add_argument("--still", type=float, default=0.003, help="changed fraction frame-to-frame below which the picture counts as still")
    ap.add_argument("--same", type=float, default=0.001, help="changed fraction vs the last saved photo at or below which it is a duplicate")
    ap.add_argument("--debug", action="store_true", help="print the motion numbers once a second")
    ap.add_argument("--note", default="", help="instruction line shown on the status page")
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        sys.exit("label: letters, digits, _ and - only")
    folder = args.out / args.label
    folder.mkdir(parents=True, exist_ok=True)

    nodes = [args.device] if args.device else find_camera_nodes(args.camera_name)
    if not nodes:
        sys.exit(f"no camera named like '{args.camera_name}' in /sys/class/video4linux")
    cap, dev = open_camera(nodes, args.width, args.height, args.fps)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    print(f"camera {dev}: {w}x{h} {fourcc} (asked {args.width}x{args.height})", flush=True)

    shared = Shared(args.label, folder)
    shared.resolution, shared.device = f"{w}x{h}", dev
    shared.note, shared.max = args.note, args.max
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(shared))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"status page: http://127.0.0.1:{args.port}   saving to {folder}", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    # Two numbers per frame, both "fraction of pixels that changed" on a small blurred grey image:
    #   d     = this frame vs the previous one  -> is the picture still?
    #   d_ref = this frame vs the last saved one -> did a hand / the brick change the scene?
    # Arm when d_ref > --motion (a hand came in). Save when armed and still for --still-s,
    # unless the scene is back to the last saved photo (d_ref <= --same): then it is a duplicate.
    prev = baseline = None
    armed = False
    still_since = None
    last_live = last_dbg = saved_at = 0.0
    dmax = dref_max = 0.0
    fails = 0
    try:
        while not stop.is_set() and not shared.next_requested.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 30:
                    print("camera stopped sending frames", flush=True)
                    break
                time.sleep(0.05)
                continue
            fails = 0
            now = time.monotonic()
            g = small_gray(frame)
            if baseline is None:
                baseline = g  # the scene at start is the reference, it is not saved
            with shared.lock:  # reference = last photo still kept (undo goes back one)
                ref = shared.saved[-1][1] if shared.saved else baseline
            d = changed_fraction(g, prev, args.pix_t) if prev is not None else 0.0
            d_ref = changed_fraction(g, ref, args.pix_t)
            prev = g
            dmax, dref_max = max(dmax, d), max(dref_max, d_ref)
            if d_ref > args.motion:
                armed = True

            if d < args.still:
                still_since = still_since or now
                state = "waiting"
                if armed and now - still_since >= args.still_s:
                    armed = False
                    diff_saved = d_ref
                    if diff_saved > args.same:
                        t = time.time()
                        path = folder / (time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + f"-{int(t * 1000) % 1000:03d}.jpg")
                        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        with shared.lock:
                            shared.saved.append((path, g))
                            shared.count_run += 1
                            shared.count_total += 1
                            shared.last_jpg = jpeg(frame, 640)
                        saved_at = now
                        print(f"saved {path.name}  ({shared.count_run} this run, {shared.count_total} in {args.label}, "
                              f"diff vs last {diff_saved:.4f})", flush=True)
                        if args.max and shared.count_run >= args.max:
                            print(f"reached --max {args.max}", flush=True)
                            break
                    else:
                        print(f"skipped: same as the last photo (diff {diff_saved:.4f})", flush=True)
            else:
                still_since = None
                state = "motion"

            with shared.lock:
                shared.state = "saved" if now - saved_at < 1.5 and state != "motion" else state
            if now - last_live > 0.3:
                live = jpeg(frame, 640, 70)
                with shared.lock:
                    shared.live_jpg = live
                last_live = now
            if args.debug and now - last_dbg > 1.0:
                print(f"[debug] last 1 s: max frame-to-frame {dmax:.4f}, max vs last saved {dref_max:.4f}, "
                      f"armed={armed} state={state}", flush=True)
                dmax, dref_max, last_dbg = 0.0, 0.0, now
    finally:
        cap.release()
        server.shutdown()
        print(f"stopped. {shared.count_run} photos this run, {shared.count_total} in {folder}", flush=True)


if __name__ == "__main__":
    main()
