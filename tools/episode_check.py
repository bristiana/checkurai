"""Sanity-check recorded episodes (LeRobot v3 dataset dir, e.g. a Studio cache or dataset folder).

Per episode: length, start/end distance to HOME (config/home_pose.json, Studio units), how far the arm moved,
whether the gripper closed and opened (a grasp), and frozen stretches (leader dropout: action unchanged >1 s
while the arm was away from HOME).

    python tools/episode_check.py <dataset dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
d = Path(sys.argv[1])
home = json.loads((ROOT / "config/home_pose.json").read_text())
H = np.array(home["home"])
info = json.loads((d / "meta/info.json").read_text())
fps = info["fps"]
df = pd.concat(pd.read_parquet(p) for p in sorted((d / "data").rglob("*.parquet")))
print(f"{d.name}: {info['total_episodes']} episodes, {info['total_frames']} frames, {fps} fps")
print(f"{'ep':>3} {'secs':>5} {'start':>6} {'end':>6} {'moved':>6} {'grip':>9} {'frozen':>7}  verdict")
for ep, g in df.groupby("episode_index"):
    s = np.stack(g["observation.state"].to_numpy())
    a = np.stack(g["action"].to_numpy())
    secs = len(g) / fps
    d0 = np.abs(s[:15].mean(0) - H)[:5].max()
    d1 = np.abs(s[-15:].mean(0) - H)[:5].max()
    moved = np.abs(s[:, :5] - s[0, :5]).max()
    grip = a[:, 5]
    grasp = f"{grip.min():.0f}..{grip.max():.0f}"
    away = np.abs(s[:, :5] - H[:5]).max(1) > 15
    same = np.r_[False, (np.abs(np.diff(a, axis=0)).max(1) < 1e-6)]
    run = best = 0
    for f, w in zip(same, away):
        run = run + 1 if (f and w) else 0
        best = max(best, run)
    frozen = best / fps
    bad = []
    if secs < 4: bad.append("too short (double key press?)")
    if moved < 20: bad.append("arm hardly moved")
    if grip.max() - grip.min() < 10: bad.append("gripper never opened/closed")
    if d0 > 12: bad.append(f"did not start at HOME")
    if d1 > 12: bad.append(f"did not end at HOME")
    if frozen > 1.0: bad.append(f"leader froze {frozen:.1f}s")
    print(f"{ep:>3} {secs:5.1f} {d0:6.1f} {d1:6.1f} {moved:6.1f} {grasp:>9} {frozen:6.1f}s  {'OK' if not bad else 'CHECK: ' + '; '.join(bad)}")
