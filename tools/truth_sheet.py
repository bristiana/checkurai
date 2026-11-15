#!/usr/bin/env python3
"""Contact sheet for hand-labelling the closed-loop bricks: one tile per sorted cycle (the inspection crops the
cell saved), captioned with run, cycle, verdict and score, plus a CSV to fill in.

    python tools/truth_sheet.py            # -> results/truth_sheet.jpg + results/truth_todo.csv
Copy the rows you label into results/truth.csv (header run,cycle,truth; truth = good | defect)."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
cl = json.loads((ROOT / "results" / "closed_loop.json").read_text())
tiles, rows = [], []
for r in cl["rows"]:
    if r.get("result") != "sorted":
        continue
    img = cv2.imread(str(ROOT / "data" / "cell_runs" / r["run"] / f"cycle_{int(r['cycle']):03d}_inspect.jpg"))
    if img is None:
        img = np.zeros((256, 768, 3), np.uint8)
    img = cv2.resize(img, (768, 256))
    bar = np.full((40, 768, 3), 255, np.uint8)
    cap = f"#{len(tiles) + 1}  {r['run']} c{r['cycle']}  verdict {r.get('verdict')}  score {r.get('score')}"
    cv2.putText(bar, cap, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    tiles.append(np.vstack([bar, img]))
    rows.append({"n": len(tiles), "run": r["run"], "cycle": r["cycle"], "verdict": r.get("verdict"), "truth": ""})
cols = 2
while len(tiles) % cols:
    tiles.append(np.full_like(tiles[0], 255))
sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
cv2.imwrite(str(ROOT / "results" / "truth_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
with open(ROOT / "results" / "truth_todo.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["n", "run", "cycle", "verdict", "truth"])
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} bricks -> results/truth_sheet.jpg, results/truth_todo.csv")
