"""Print camera role -> /dev/videoN, found by USB name (survives recabling). Usage: python find_cams.py [--json]"""
import glob, json, os, sys
ROLES = {"top": "RealSense", "wrist": "USB2.0_CAM1", "inspection": "Webcam Audio"}
out = {}
for d in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(p.rsplit("video", 1)[1])):
    name = open(d + "/name").read().strip()
    if int(open(d + "/index").read()) != 0: continue          # index 0 = the capture node
    intf = os.path.realpath(d + "/device").rsplit(":", 1)[-1]
    for role, key in ROLES.items():
        if key in name and role not in out and not (role == "top" and intf != "1.3"):  # D415 RGB = USB interface 1.3
            out[role] = "/dev/" + os.path.basename(d)
if "--json" in sys.argv: print(json.dumps(out))
else:
    for r in ROLES: print(f"{r:11s} {out.get(r, 'NOT FOUND')}  ({ROLES[r]}, {'YUYV' if r == 'top' else 'MJPG'})")
