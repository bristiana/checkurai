"""Does a place policy steer towards where the bowl actually is (from the top camera), or replay a fixed motion?

Counterfactual test on the demos: for a demo shortly after the grasp, run the trained policy (its untouched
export, act.orig.xml) on that demo's arm state + wrist picture, once with its own top picture and once with the
top picture of another demo (same moment) where the bowl stood elsewhere. If the predicted pan over the bowl
moves with the swapped bowl, the policy follows the bowl by itself; if not, it needs a motion correction.

    ~/physical-ai-studio/application/backend/.venv/bin/python tools/bowl_react.py act-place-bad-v3 orange
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy_tune as PT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def lb(img_bgr: np.ndarray) -> np.ndarray:
    return PT.net_input(img_bgr, PT.ACT_GEO)


def main() -> int:
    name, color = sys.argv[1], sys.argv[2]
    rec = PT.model_record(name)
    d = PT.export_dir(rec)
    xml = d / ("act.orig.xml" if (d / "act.orig.xml").exists() else "act.xml")
    model = ov.Core().compile_model(xml, "CPU")
    snap = PT.snapshot_dir(rec)
    bf = json.loads((ROOT / "config" / "bowl_follow.json").read_text())
    demos = {r["ep"]: r for r in bf["demos"] if r["color"] == color}
    eps = pd.concat([pd.read_parquet(f) for f in sorted((snap / "meta" / "episodes").rglob("*.parquet"))])
    data = pd.concat([pd.read_parquet(f) for f in sorted((snap / "data").rglob("*.parquet"))])

    def obs(ep: int, frame: int) -> dict:
        e = eps[eps.episode_index == ep].iloc[0]
        d_ = data[(data.episode_index == ep) & (data.frame_index == frame)].iloc[0]
        out = {"state": np.asarray(d_["observation.state"], np.float32)[None]}
        for cam in ("top", "wrist"):
            key = f"observation.images.{cam}"
            v = snap / "videos" / key / f"chunk-{int(e[f'videos/{key}/chunk_index']):03d}" / \
                f"file-{int(e[f'videos/{key}/file_index']):03d}.mp4"
            f = PT.video_frame(v, float(e[f"videos/{key}/from_timestamp"]) + float(d_["timestamp"]))
            out[f"images.{cam}"] = lb(f)
        return out

    def release_pan(o: dict) -> float:
        a = model(o)[0][0]  # [chunk, 6]
        g = a[:, 5]
        k = int(np.argmax(g > 20)) if (g > 20).any() else len(g) - 1  # where the predicted gripper opens
        return float(a[max(k - 3, 0), 0]), int(k)

    # demos in the snapshot with a found release, grouped by bowl position
    have = sorted(set(demos) & set(eps.episode_index.astype(int)))
    rows = []
    for ep in have:
        r = demos[ep]
        fr = r["grasp"] + 20  # just after lifting the brick
        rows.append((ep, r["x"], r["y"], r["pan"], obs(ep, fr)))
    print(f"{name}: {len(rows)} demos; predicted pan at release (own top picture) vs the demo's real release pan")
    own = []
    for ep, x, y, pan, o in rows:
        p, k = release_pan(o)
        own.append(p)
        print(f"  ep {ep:2d} bowl ({x:.0f},{y:.0f}) real {pan:6.1f} predicted {p:6.1f} (opens at step {k})")
    # swap: every demo's state/wrist with top pictures from demos whose bowl stood far away
    print("swapped top pictures: change of predicted pan vs change the demos show for that bowl move")
    gain = []
    for i, (ep, x, y, pan, o) in enumerate(rows):
        far = max(range(len(rows)), key=lambda j: abs(rows[j][2] - y) + abs(rows[j][1] - x))
        ep2, x2, y2, pan2, o2 = rows[far]
        if abs(y2 - y) + abs(x2 - x) < 25:
            continue
        sw = dict(o)
        sw["images.top"] = o2["images.top"]
        p_sw, _ = release_pan(sw)
        expect = pan2 - pan  # how the demonstrator's pan differed between the two bowl positions
        moved = p_sw - own[i]
        gain.append(moved / expect if abs(expect) > 3 else np.nan)
        print(f"  ep {ep:2d} + top of ep {ep2:2d} (bowl {y:.0f}->{y2:.0f}): predicted pan moved {moved:+6.1f}, "
              f"demos differ by {expect:+6.1f}")
    g = np.array([v for v in gain if np.isfinite(v)])
    if len(g):
        print(f"policy follows the bowl by {np.median(g) * 100:.0f} % of the demonstrated pan change (median of {len(g)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
