"""How does the arm's release pose depend on where the bowl stands? (from the place demos)

For every place demo: the top camera's first frame gives the bowl position, mapped onto one fixed table frame
(the markers of config/cell_ref/top_place_demo.png), and the joint state when the gripper opens over the bowl
is the release pose. A fit of release pose vs bowl position tells how much each joint must change when a bowl
stands elsewhere - the basis for shifting a policy's motion towards a moved bowl.

    ~/physical-ai-studio/application/backend/.venv/bin/python tools/bowl_follow.py analyze
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy_tune as PT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CANON = ROOT / "config" / "cell_ref" / "top_place_demo.png"
DATASETS = {"lilac": "50677384-1d74-4183-aef4-235843eab71d", "orange": "16871ef1-0d3d-4e62-bd42-e5d8de3c204b"}
JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
OUT = ROOT / "config" / "bowl_follow.json"


def canon_markers() -> dict[int, np.ndarray]:
    return PT.detect(cv2.imread(str(CANON)))


def release_pose(s: np.ndarray, a: np.ndarray) -> tuple[int, int] | None:
    """(grasp frame, release frame): the gripper command closes on the brick after having opened, then opens
    again later, with the arm turned well away from the spot (a re-grasp at the spot does not count)."""
    g = a[:, 5]
    n = len(g)
    opened = np.where(g > 20)[0]
    if not len(opened):
        return None
    i = opened[0]
    while i < n - 10 and not (g[i] < 18 and (g[i:i + 10] < 18).all()):
        i += 1
    if i >= n - 10:
        return None
    grasp = i
    j = grasp + 15
    while j < n and not (g[j] >= 20 and g[j] > g[grasp:j].min() + 6 and abs(s[j, 0] - s[grasp, 0]) > 25):
        j += 1
    return (grasp, j) if j < n else None


def analyze() -> dict:
    canon = canon_markers()
    rows = []
    for color, ds in DATASETS.items():
        root = PT.STUDIO / "datasets" / ds
        eps = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))])
        data = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet"))])
        for _, e in eps.sort_values("episode_index").iterrows():
            ep = int(e["episode_index"])
            v = root / "videos" / PT.TOP / f"chunk-{int(e[f'videos/{PT.TOP}/chunk_index']):03d}" / \
                f"file-{int(e[f'videos/{PT.TOP}/file_index']):03d}.mp4"
            f = PT.video_frame(v, float(e[f"videos/{PT.TOP}/from_timestamp"]) + 0.3)
            if f is None:
                continue
            try:
                w = PT.fit_warp(PT.detect(f), canon)
            except SystemExit:
                continue
            b = PT.find_bowl(f, color)
            if b is None:
                continue
            p = cv2.perspectiveTransform(b[0][None, None], np.array(w["H"]))[0, 0]
            d = data[data.episode_index == ep].sort_values("frame_index")
            s, a = np.stack(d["observation.state"].values), np.stack(d["action"].values)
            gr = release_pose(s, a)
            if gr is None:
                print(f"{color} ep {ep}: no grasp/release found")
                continue
            rows.append({"color": color, "ep": ep, "x": float(p[0]), "y": float(p[1]), "grasp": int(gr[0]),
                         "release": int(gr[1]), "grasp_pan": float(s[gr[0], 0]),
                         **{j: float(s[gr[1], k]) for k, j in enumerate(JOINTS)}})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(1).to_string(index=False))
    # per bowl: release pose ~ a + b*x + c*y (least squares), and leave-one-out error vs just the mean
    fits = {}
    for color, g in df.groupby("color"):
        X = np.c_[np.ones(len(g)), g.x - g.x.mean(), g.y - g.y.mean()]
        out = {"x0": float(g.x.mean()), "y0": float(g.y.mean()), "n": len(g),
               "x_range": [float(g.x.min()), float(g.x.max())], "y_range": [float(g.y.min()), float(g.y.max())]}
        print(f"\n{color}: {len(g)} demos, bowl x {g.x.min():.0f}..{g.x.max():.0f}, y {g.y.min():.0f}..{g.y.max():.0f}")
        for j in JOINTS[:5]:
            yv = g[j].values
            coef = np.linalg.lstsq(X, yv, rcond=None)[0]
            loo, base = [], []
            for k in range(len(g)):
                m = np.ones(len(g), bool)
                m[k] = False
                c = np.linalg.lstsq(X[m], yv[m], rcond=None)[0]
                loo.append(abs(X[k] @ c - yv[k]))
                base.append(abs(yv[m].mean() - yv[k]))
            out[j] = {"a": float(coef[0]), "per_px_x": float(coef[1]), "per_px_y": float(coef[2]),
                      "loo_err": float(np.mean(loo)), "mean_only_err": float(np.mean(base))}
            print(f"  {j:6s} = {coef[0]:7.1f} {coef[1]:+.3f}*dx {coef[2]:+.3f}*dy   leave-one-out error {np.mean(loo):5.1f} "
                  f"(vs {np.mean(base):5.1f} ignoring the bowl)")
        fits[color] = out
    base = fit_base(df, canon)
    OUT.write_text(json.dumps({"canon": str(CANON.name), "fits": fits, "base": base, "demos": rows}, indent=1) + "\n")
    print(f"\nsaved {OUT}")
    return fits


def spot_xy(canon: dict[int, np.ndarray]) -> np.ndarray:
    """The inspection spot in the canonical picture: between markers 4 and 5, a little towards the robot."""
    m4, m5 = canon[4].mean(0), canon[5].mean(0)
    return (m4 + m5) / 2 + np.array([-6.0, -13.0])


def fit_base(df: pd.DataFrame, canon: dict[int, np.ndarray]) -> dict:
    """pan = c + k * angle(bowl - base): where the robot base is in the picture, and pan units per degree.
    Data: the release pans over the bowls and the grasp pans at the spot."""
    from scipy.optimize import least_squares

    sp = spot_xy(canon)
    pts = [(r.x, r.y, r.pan) for r in df.itertuples()] + [(sp[0], sp[1], r.grasp_pan) for r in df.itertuples()]
    P = np.array(pts)

    def ang(b):
        return np.degrees(np.arctan2(P[:, 0] - b[0], P[:, 1] - b[1]))

    def res(v):
        return v[2] + v[3] * ang(v[:2]) - P[:, 2]

    best = None
    for bx in (200, 350, 500):
        for by in (-300, -100, 0):
            for k in (1.0, -1.0):
                r = least_squares(res, [bx, by, 0.0, k], loss="soft_l1", f_scale=5.0)
                if best is None or r.cost < best.cost:
                    best = r
    v = best.x
    e = np.abs(res(v))
    nb = len(df)
    print(f"robot base in the picture ~({v[0]:.0f}, {v[1]:.0f}) px, pan = {v[2]:.1f} + {v[3]:.3f} * angle(deg); "
          f"median |error| over the bowls {np.median(e[:nb]):.1f}, at the spot {np.median(e[nb:]):.1f} pan units")
    for c in sorted(set(df.color)):
        idx = np.where(df.color.values == c)[0]
        print(f"  {c}: median |error| {np.median(e[idx]):.1f}")
    return {"bx": float(v[0]), "by": float(v[1]), "c": float(v[2]), "k": float(v[3]), "spot": sp.tolist(),
            "median_err_bowls": float(np.median(e[:nb]))}


if __name__ == "__main__":
    analyze()
