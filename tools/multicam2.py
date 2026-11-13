import cv2, time, glob, os, threading
devs = sorted(set(os.path.realpath(p) for p in glob.glob('/dev/v4l/by-id/*index0')))
def open_cam(d):
    c = cv2.VideoCapture(d, cv2.CAP_V4L2)
    if 'RealSense' not in open(f'/sys/class/video4linux/{os.path.basename(d)}/name').read():  # USB webcams: MJPG keeps USB2 bandwidth low
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    c.set(3, 640); c.set(4, 480); c.set(5, 30)
    return c
def run(d, secs, out):
    c = open_cam(d); [c.read() for _ in range(5)]
    n = 0; t0 = time.time()
    while time.time() - t0 < secs:
        ok, f = c.read(); n += ok
    fourcc = int(c.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode(errors='ignore')
    out[d] = (n / secs, fourcc); c.release()
for d in devs:
    r = {}; run(d, 3, r); print(f"alone   {d}: {r[d][0]:.1f} fps ({r[d][1]})")
r = {}; ts = [threading.Thread(target=run, args=(d, 5, r)) for d in devs]
[t.start() for t in ts]; [t.join() for t in ts]
for d in devs: print(f"all-3   {d}: {r[d][0]:.1f} fps ({r[d][1]})")
