import cv2, time, glob, os
devs = sorted(set(os.path.realpath(p) for p in glob.glob('/dev/v4l/by-id/*index0')))
caps = {}
for d in devs:
    c = cv2.VideoCapture(d, cv2.CAP_V4L2)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')); c.set(3, 640); c.set(4, 480); c.set(5, 30)
    caps[d] = c
n = {d: 0 for d in caps}; t0 = time.time()
while time.time() - t0 < 4:
    for d, c in caps.items():
        ok, f = c.read()
        if ok and f is not None: n[d] += 1
for d in caps:
    print(f"{d}: {n[d]/4:.1f} fps {'OK' if n[d] > 20 else 'PROBLEM'}")
    caps[d].release()
