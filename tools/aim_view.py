"""Live camera re-aim page: each camera's live view with a faint ghost of where it must look.

Green circles = where each ArUco marker must be; red dots = where it is now; the arrow shows which way the
marker has to travel in the picture. Nudge the camera and watch the arrows: if they get shorter you are
going the right way. A camera is done when every circle turns solid green (within 4 px).

    ~/physical-ai-studio/application/backend/.venv/bin/python tools/aim_view.py      # open http://localhost:8767

Top camera target = the first place_good demo frame (config/cell_ref/top_place_demo.png): its bowls are where
the place policies learned them. Inspection target = config/cell_ref/inspection.png (same view as the anomaly photos).
The top camera no longer has to be exact: tools/policy_tune.py --cell --top-warp auto undoes a moved camera in
software (re-run it after touching the camera); aiming close just keeps the picture edges in view.

Third panel (BOWLS): where each bowl must stand for the policy that uses it (tools/bowl_check.py).
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cam_lock as L  # noqa: E402
from cell_check import markers  # noqa: E402

try:
    import bowl_check as BC  # noqa: E402  (needs Studio's venv: openvino + pandas)
    import policy_tune as PT  # noqa: E402
except Exception as _e:  # noqa: BLE001 - the page still works without the bowl panel
    BC = PT = None
    BOWL_ERR = str(_e)

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / "config" / "cell_ref"
TARGET = {"top": REF_DIR / "top_place_demo.png", "inspection": REF_DIR / "inspection.png"}
if len(sys.argv) > 1 and Path(sys.argv[1]).exists():  # another top target, e.g. the pick demos' view
    TARGET["top"] = Path(sys.argv[1])
OK_PX = 4.0
PORT = 8767

state: dict[str, dict] = {}
lock = threading.Lock()


def hint(live: dict, target: dict) -> tuple[str, list[str]]:
    common = sorted(set(live) & set(target))
    missing = sorted(set(target) - set(live))
    lines = []
    if missing:
        lines.append(f"markers not visible: {missing} (camera pointing too far off, or something covers them)")
    if len(common) < 2:
        return "NOT FOUND", lines + ["fewer than 2 markers visible: point the camera roughly at the target picture first"]
    src = np.array([live[i]["c"] for i in common], np.float32)
    dst = np.array([target[i]["c"] for i in common], np.float32)
    err = np.linalg.norm(src - dst, axis=1)
    if err.max() <= OK_PX and not missing:
        return "ALIGNED", ["all markers within 4 px - done, tighten the camera now"]
    d = (dst - src).mean(0)  # how the picture content must travel; the camera points the opposite way
    parts = []
    if abs(d[0]) > 2:
        parts.append(f"{'RIGHT' if d[0] < 0 else 'LEFT'} side ({abs(d[0]):.0f} px)")
    if abs(d[1]) > 2:
        parts.append(f"{'BOTTOM' if d[1] < 0 else 'TOP'} ({abs(d[1]):.0f} px)")
    if parts:
        lines.append("point the camera a bit more towards the " + " and the ".join(parts) +
                     " of this picture - small nudges, then check the arrows got shorter")
    if len(common) >= 3:
        m, _ = cv2.estimateAffinePartial2D(src, dst)
        if m is not None:
            ang = float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
            scale = float(np.hypot(m[0, 0], m[1, 0]))
            if abs(ang) > 0.4:
                lines.append(f"the picture is also turned ~{abs(ang):.1f} deg: it must turn "
                             f"{'clockwise' if ang > 0 else 'counter-clockwise'}. Twist the camera a little; "
                             "if the arrows get longer, twist the other way")
            if abs(scale - 1) > 0.02:
                lines.append("markers look " + ("too small: camera is too far / higher than before - bring it closer"
                                                if scale > 1 else "too big: camera is too close / lower than before - move it back"))
    return f"OFF by up to {err.max():.0f} px", lines


def bowls(f: np.ndarray, targets) -> None:
    if BC is None:
        with lock:
            state["bowls"] = {"status": "NO BOWL CHECK", "lines": [BOWL_ERR[:200]], "jpg": None}
        return
    try:
        live = PT.detect(f)
        img, lines, ok = BC.check(f, live, targets.get())
        status = "ALIGNED all bowls" if ok else "OFF - slide the bowls along the arrows"
    except BaseException as e:  # noqa: BLE001 - keep the page alive and say why
        img, lines, status = f, [str(e)[:200]], "ERROR"
    okj, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with lock:
        state["bowls"] = {"status": status, "lines": lines, "jpg": jpg.tobytes() if okj else None}


def worker(role: str) -> None:
    ghost = cv2.imread(str(TARGET[role]))
    target = markers(cv2.cvtColor(ghost, cv2.COLOR_BGR2GRAY))
    targets = BC.Targets() if (BC is not None and role == "top") else None
    last_bowls = 0.0
    while True:
        try:
            if role not in L.SHARED:
                L.open_source(role)
            f = L.frames(role, 1, gap=0.0)[-1]
        except Exception as e:  # noqa: BLE001 - keep the page alive and say why
            with lock:
                state[role] = {"status": "NO CAMERA", "lines": [str(e)[:200]], "jpg": state.get(role, {}).get("jpg")}
            time.sleep(1)
            continue
        if f.shape[:2] != ghost.shape[:2]:
            f = cv2.resize(f, (ghost.shape[1], ghost.shape[0]))
        if role == "top" and time.time() - last_bowls > 0.7:
            bowls(f, targets)
            last_bowls = time.time()
        live = markers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        img = cv2.addWeighted(f, 0.6, ghost, 0.4, 0)
        for i, t in target.items():
            tc = tuple(int(v) for v in t["c"])
            ok = i in live and np.linalg.norm(np.subtract(live[i]["c"], t["c"])) <= OK_PX
            cv2.circle(img, tc, 14, (0, 200, 0), -1 if ok else 3)
            cv2.putText(img, str(i), (tc[0] + 16, tc[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 160, 0), 2)
            if i in live and not ok:
                lc = tuple(int(v) for v in live[i]["c"])
                cv2.circle(img, lc, 6, (0, 0, 255), -1)
                cv2.arrowedLine(img, lc, tc, (0, 0, 255), 3, tipLength=0.25)
        status, lines = hint(live, target)
        if role == "top" and not status.startswith("ALIGNED") and status != "NOT FOUND":
            lines.append("the policies get this undone in software (policy_tune.py --cell --top-warp auto, re-run "
                         "after touching the camera); aiming closer only keeps the picture edges in view")
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with lock:
            state[role] = {"status": status, "lines": lines, "jpg": jpg.tobytes() if ok else None}
        time.sleep(0.15)


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Camera re-aim</title><style>
body{font-family:sans-serif;background:#111;color:#eee;margin:12px}
.cam{display:inline-block;vertical-align:top;margin:0 12px 12px 0;max-width:660px}
img{width:640px;max-width:100%;border:2px solid #444}
.s{font-size:22px;font-weight:bold;margin:6px 0}.ALIGNED{color:#3c3}.OFF{color:#f84}
li{margin:3px 0;font-size:16px}</style></head><body>
<p>Green circle = where the marker must be. Red dot + arrow = where it is now and which way it must travel in the
picture. Nudge the camera a little, wait 1 s, and check that the arrows get shorter. Done when every circle is solid green.</p>
<div class=cam><h2>TOP camera</h2><div id=s_top class=s></div><img id=i_top><ul id=l_top></ul></div>
<div class=cam><h2>INSPECTION camera</h2><div id=s_inspection class=s></div><img id=i_inspection><ul id=l_inspection></ul></div>
<div class=cam><h2>BOWLS (top camera)</h2><p>Dashed circle = where the policy saw the bowl in its demos, thin circle =
the OK zone for the bowl centre, solid = where it is now. Slide the bowl along the arrow, then tape it down.</p>
<div id=s_bowls class=s></div><img id=i_bowls><ul id=l_bowls></ul></div>
<script>
async function tick(){
  try{const r=await fetch('/status.json');const d=await r.json();
  for(const k of ['top','inspection','bowls']){const v=d[k]||{status:'starting...',lines:[]};
    const s=document.getElementById('s_'+k);s.textContent=v.status;s.className='s '+v.status.split(' ')[0];
    document.getElementById('l_'+k).innerHTML=v.lines.map(x=>'<li>'+x+'</li>').join('');
    document.getElementById('i_'+k).src='/'+k+'.jpg?'+Date.now();}}catch(e){}
  setTimeout(tick,500);}
tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/":
            body, ctype = PAGE.encode(), "text/html"
        elif path == "/status.json":
            with lock:
                body = json.dumps({k: {"status": v["status"], "lines": v["lines"]} for k, v in state.items()}).encode()
            ctype = "application/json"
        elif path in ("/top.jpg", "/inspection.jpg", "/bowls.jpg"):
            with lock:
                body = (state.get(path[1:-4]) or {}).get("jpg") or b""
            ctype = "image/jpeg"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    for role in TARGET:
        threading.Thread(target=worker, args=(role,), daemon=True).start()
    print(f"camera re-aim page on http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
