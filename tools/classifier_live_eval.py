"""Score the live cycles with a trained classifier (models/classifier/NAME): full frame -> the exact pipeline
(tools/brick_classifier.py); saved montage -> its 3 old-pipeline tiles, label inpainted, grey fill by colour hull."""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "4"
import sys, glob, csv, json
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/home/intel/checkurai")
sys.path.insert(0, "/tmp")
from tools.brick_classifier import BrickClassifierInspector
def hull_fill(t, value=128):
    hsv = cv2.cvtColor(cv2.GaussianBlur(t, (5, 5), 0), cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32); cand = s >= 80
    if cand.sum() < 200: return t
    col = (cand & (s >= 0.6 * np.percentile(s[cand], 90))).astype(np.uint8)
    col = cv2.morphologyEx(col, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(col)
    if n <= 1: return t
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= max(30, 0.25 * big)]
    m = np.zeros(t.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(cv2.findNonZero(np.isin(lab, keep).astype(np.uint8))), 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    out = t.copy(); out[~m] = value; return out

def label_mask(m):
    lab = np.zeros(m.shape[:2], np.uint8); roi = m[:60, :256].astype(np.int16)
    for c in ((60, 180, 60), (40, 60, 220)):
        lab[:60, :256] |= (np.abs(roi - np.array(c)).max(axis=2) < 60).astype(np.uint8)
    return cv2.dilate(lab, np.ones((7, 7), np.uint8)) > 0


R = Path("/home/intel/checkurai")
name = sys.argv[1]
dev = sys.argv[2] if len(sys.argv) > 2 else "CPU"
thr_override = float(sys.argv[3]) if len(sys.argv) > 3 else None
ins = BrickClassifierInspector(name, dev)
thr = thr_override if thr_override is not None else ins.low
truth = {(r["run"], int(r["cycle"])): r["truth"] for r in csv.DictReader(open(R / "results/truth.csv"))}
DEFECT = {("20261114-202926", 0), ("20261114-203554", 0)}
rows = []
for f in sorted(glob.glob(str(R / "data/cell_runs/2026111[45]-*/cycle_*_inspect.jpg"))):
    run, ci = f.split("/")[-2], int(f.split("/")[-1].split("_")[1])
    lab = "defect" if (run, ci) in DEFECT else truth.get((run, ci), "good")
    full = f.replace("_inspect.jpg", "_frame.jpg")
    if os.path.exists(full):
        r = ins.inspect([cv2.imread(full)])
        p, how = list(ins.last_views.values()), "frame"
    else:
        m = cv2.imread(f)
        lm = label_mask(m)[:, :256]
        tiles = [m[:, i * 256:(i + 1) * 256].copy() for i in range(3)]
        tiles[0] = cv2.inpaint(tiles[0], lm.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        p, how = list(ins.predict([hull_fill(t) for t in tiles])), "montage"
    s = max(p)
    rows.append((run, ci, lab, round(float(s), 3), [round(float(x), 3) for x in p], how))
print(f"{name}: threshold {thr:.3f}")
for r in rows:
    print(" ", r[0], f"c{r[1]}", r[2], r[3], "DEFECT" if r[3] >= thr else "good", r[4], r[5])
cl = [r for r in rows if r[2] != "defect"]
df = [r for r in rows if r[2] == "defect"]
print(f"LIVE clean called good {sum(r[3] < thr for r in cl)}/{len(cl)} | defects caught {sum(r[3] >= thr for r in df)}/{len(df)}")
