"""CheckurAI live camera dashboard.

Usage:  python camdash.py [--port 8080] [--host 127.0.0.1]
View:   on your own laptop run  ssh -N -L 8080:localhost:8080 intel  then open http://localhost:8080

Cameras are opened only while someone is watching (released ~3 s after the last viewer leaves),
so the dashboard does not block lerobot / Physical AI Studio when nobody has it open.
Snapshots go to ~/checkurai/snapshots/<role>/ (handy for the Anomalib dataset).
"""
import argparse, glob, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

ROLES = {  # role -> (substring of V4L2 name, required USB interface or None, label)
    "top": ("RealSense", "1.3", "Top view (RealSense D415)"),
    "inspection": ("Webcam Audio", None, "Inspection"),
    "wrist": ("USB2.0_CAM1", None, "Wrist"),
}
W, H, FPS, JPEG_Q, IDLE_S = 640, 480, 30, 80, 3.0
SNAP_DIR = os.path.expanduser("~/checkurai/snapshots")


def find_devices():
    out = {}
    for d in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(p.rsplit("video", 1)[1])):
        try:
            name = open(d + "/name").read().strip()
            if int(open(d + "/index").read()) != 0:
                continue
            intf = os.path.realpath(d + "/device").rsplit(":", 1)[-1]
        except OSError:
            continue
        for role, (key, need_intf, _) in ROLES.items():
            if key in name and role not in out and (need_intf is None or intf == need_intf):
                out[role] = "/dev/" + os.path.basename(d)
    return out


class Camera:
    def __init__(self, role, dev):
        self.role, self.dev, self.label = role, dev, ROLES[role][2]
        self.cond = threading.Condition()
        self.jpeg, self.seq, self.viewers = None, 0, 0
        self.last_want = 0.0
        self.fps, self.err, self.running, self.size = 0.0, None, False, None
        self.thread = None
        self.lock = threading.Lock()

    def want(self):
        self.last_want = time.time()
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._loop, daemon=True)
                self.thread.start()

    def _open(self):
        c = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        if "RealSense" not in ROLES[self.role][0]:
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # keep USB2 bandwidth low
        c.set(cv2.CAP_PROP_FRAME_WIDTH, W); c.set(cv2.CAP_PROP_FRAME_HEIGHT, H); c.set(cv2.CAP_PROP_FPS, FPS)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return c

    def _loop(self):
        cap = None
        n, t0 = 0, time.time()
        try:
            cap = self._open()
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {self.dev} (busy? another program may be using it)")
            self.err = None
            fails = 0
            while self.viewers > 0 or time.time() - self.last_want < IDLE_S:
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails > 30:
                        raise RuntimeError(f"no frames from {self.dev}")
                    time.sleep(0.02); continue
                fails = 0
                self.size = (frame.shape[1], frame.shape[0])
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                if not ok:
                    continue
                with self.cond:
                    self.jpeg, self.raw = buf.tobytes(), frame
                    self.seq += 1
                    self.cond.notify_all()
                n += 1
                if time.time() - t0 >= 1.0:
                    self.fps, n, t0 = n / (time.time() - t0), 0, time.time()
        except Exception as e:  # noqa: BLE001
            self.err = str(e)
        finally:
            if cap is not None:
                cap.release()
            self.fps = 0.0
            with self.lock:
                self.running = False
            with self.cond:
                self.cond.notify_all()

    def frames(self):
        last = -1
        while True:
            self.want()
            with self.cond:
                self.cond.wait_for(lambda: (self.seq != last and self.jpeg) or self.err, timeout=2.0)
                if self.seq == last or not self.jpeg:
                    if self.err:
                        time.sleep(1.0)  # retry slowly while device is busy/missing
                    continue
                last, data = self.seq, self.jpeg
            yield data

    def latest_jpeg(self, timeout=3.0):
        """Most recent JPEG (waits for the first one if the camera was idle)."""
        self.want()
        with self.cond:
            self.cond.wait_for(lambda: (self.running and self.jpeg) or self.err, timeout=timeout)
            return self.jpeg if self.running else None

    def latest_raw(self, timeout=3.0):
        self.want()
        start = self.seq
        with self.cond:
            self.cond.wait_for(lambda: self.seq != start, timeout=timeout)
            return getattr(self, "raw", None) if self.seq != start else None


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CheckurAI Cameras</title><style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e7e9ee;--mut:#8a90a0;--ok:#3ecf8e;--bad:#ff6b6b;--acc:#5b8cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid #262a33}
h1{font-size:16px;margin:0}.mut{color:var(--mut)}
main{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:12px 16px}
@media (max-width:1100px){main{grid-template-columns:repeat(2,1fr)}}
@media (max-width:700px){main{grid-template-columns:1fr}}
.card{background:var(--card);border-radius:10px;overflow:hidden;border:1px solid #262a33}
.bar{display:flex;align-items:center;gap:8px;padding:8px 10px}
.bar b{flex:1}.dot{width:8px;height:8px;border-radius:50%;background:var(--mut)}
.dot.ok{background:var(--ok)}.dot.bad{background:var(--bad)}
.vid{position:relative;aspect-ratio:4/3;background:#000}
.vid img{width:100%;height:100%;object-fit:contain;display:block}
.vid .err{position:absolute;inset:0;display:none;align-items:center;justify-content:center;padding:16px;text-align:center;color:var(--bad)}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:5px 10px;cursor:pointer;font:inherit}
button.sec{background:#2a2f3a}
.toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#2a2f3a;padding:8px 14px;border-radius:8px;opacity:0;transition:.2s}
</style></head><body>
<header><h1>CheckurAI · cameras</h1><span class="mut" id="clock"></span></header>
<main id="grid"></main><div class="toast" id="toast"></div>
<script>
const cams = __CAMS__;
const grid = document.getElementById('grid');
// 2 lanes x 3 cameras = 6 connections = the browser's per-host HTTP/1.1 limit.
const LANES = 2, view = {};
for (const c of cams) {
  grid.insertAdjacentHTML('beforeend', `<div class="card"><div class="bar"><span class="dot" id="d-${c.role}"></span>
  <b>${c.label}</b><span class="mut" id="s-${c.role}">${c.dev}</span>
  <button class="sec" onclick="full('${c.role}')">⤢</button><button onclick="snap('${c.role}')">Snapshot</button></div>
  <div class="vid"><img id="i-${c.role}" alt="${c.label}"><div class="err" id="e-${c.role}"></div></div></div>`);
  pull(c.role);
}
// Pull loops: each lane requests the next frame only after its previous one arrived -> lag never builds up.
// LANES requests in flight per camera hide the Wi-Fi round trip; frames arriving out of order are dropped.
function pull(r){
  view[r] = {n:0, t0:performance.now(), fps:0, ms:0, shown:0, prev:null};
  for (let k = 0; k < LANES; k++) setTimeout(() => lane(r), k * 60);
}
async function lane(r){
  const img = document.getElementById('i-'+r), v = view[r];
  while (true) {
    const t = performance.now();
    try {
      const res = await fetch('/frame/'+r+'?t='+t, {cache:'no-store'});
      if (!res.ok) throw 0;
      const blob = await res.blob();
      if (t < v.shown) continue;            // an newer frame is already on screen
      v.shown = t;
      const url = URL.createObjectURL(blob);
      img.src = url; if (v.prev) URL.revokeObjectURL(v.prev); v.prev = url;
      v.ms = performance.now()-t; v.n++;
      if (performance.now()-v.t0 > 1000) { v.fps = v.n*1000/(performance.now()-v.t0); v.n = 0; v.t0 = performance.now(); }
    } catch (_) { await new Promise(s=>setTimeout(s,1000)); }
  }
}
function toast(t){const e=document.getElementById('toast');e.textContent=t;e.style.opacity=1;setTimeout(()=>e.style.opacity=0,1800)}
async function snap(r){const j=await (await fetch('/snapshot/'+r,{method:'POST'})).json();toast(j.ok?('Saved '+j.path):('Error: '+j.error))}
function full(r){document.getElementById('i-'+r).requestFullscreen()}
async function poll(){
  try{const st=await (await fetch('/stats')).json();
    for(const c of cams){const s=st[c.role]||{};const d=document.getElementById('d-'+c.role),e=document.getElementById('e-'+c.role);
      d.className='dot '+(s.err?'bad':(s.fps>0?'ok':''));
      const v=view[c.role]||{};
      document.getElementById('s-'+c.role).textContent=`cam ${(s.fps||0).toFixed(0)} fps · you ${(v.fps||0).toFixed(0)} fps · ${Math.round(v.ms||0)} ms`;
      e.style.display=s.err?'flex':'none';e.textContent=s.err||'';
      if(s.err&&!s.fps){} }
  }catch(_){}
  document.getElementById('clock').textContent=new Date().toLocaleTimeString();
}
setInterval(poll,1000);poll();
</script></body></html>"""


def make_handler(cams):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: one connection per browser lane, no reconnect per frame

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                info = [{"role": r, "label": c.label, "dev": c.dev} for r, c in cams.items()]
                b = PAGE.replace("__CAMS__", json.dumps(info)).encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            elif p == "/stats":
                self._json({r: {"fps": round(c.fps, 1), "err": c.err, "size": c.size, "viewers": c.viewers,
                                "running": c.running} for r, c in cams.items()})
            elif p.startswith("/frame/") and p[7:] in cams:
                jpg = cams[p[7:]].latest_jpeg()
                if not jpg:
                    return self._json({"error": cams[p[7:]].err or "no frame"}, 503)
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(jpg)))
                self.end_headers(); self.wfile.write(jpg)
            elif p.startswith("/stream/") and p[8:] in cams:
                cam = cams[p[8:]]
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close")
                self.close_connection = True; self.end_headers()
                cam.viewers += 1
                try:
                    for jpg in cam.frames():
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:  # noqa: BLE001
                    print(f"stream {cam.role} error: {e!r}", flush=True)
                finally:
                    cam.viewers -= 1
            else:
                self.send_error(404)

        def do_POST(self):
            p = self.path.split("?")[0]
            if p.startswith("/snapshot/") and p[10:] in cams:
                role = p[10:]
                frame = cams[role].latest_raw()
                if frame is None:
                    return self._json({"ok": False, "error": cams[role].err or "no frame"}, 503)
                d = os.path.join(SNAP_DIR, role); os.makedirs(d, exist_ok=True)
                path = os.path.join(d, time.strftime("%H%M%S") + f"_{int(time.time()*1000)%1000:03d}.png")
                cv2.imwrite(path, frame)
                return self._json({"ok": True, "path": path})
            self.send_error(404)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    devs = find_devices()
    cams = {r: Camera(r, devs[r]) for r in ROLES if r in devs}
    for r in ROLES:
        print(f"{r:11s} {devs.get(r, 'NOT FOUND')}")
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(cams))
    srv.daemon_threads = True
    print(f"Dashboard on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
