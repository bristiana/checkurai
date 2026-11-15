"""How the top-camera scene changed over the demos: camera position and where each bowl stood, per episode.

For every episode of the given Studio datasets (or a model's training snapshot): the first frame's ArUco
markers give the camera offset from the live camera, and the bowl positions are mapped into today's live
picture (so camera moves and bowl moves are told apart). Bowl positions are in live-camera pixels; a bowl
radius is ~80 px (~9 cm).

    ~/physical-ai-studio/application/backend/.venv/bin/python tools/scene_history.py [dataset-id|model ...]
    (default: the pick / place_good / place_bad datasets of the models in config/cell.json)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy_tune as PT  # noqa: E402
from bowl_check import BOWLS, find_bowl  # noqa: E402  (find_bowl = policy_tune.find_bowl)

TOP = PT.TOP


def episodes(root: Path) -> pd.DataFrame:
    eps = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))])
    return eps.sort_values("episode_index")


def frame(root: Path, e) -> np.ndarray | None:
    v = root / "videos" / TOP / f"chunk-{int(e[f'videos/{TOP}/chunk_index']):03d}" / f"file-{int(e[f'videos/{TOP}/file_index']):03d}.mp4"
    return PT.video_frame(v, float(e[f"videos/{TOP}/from_timestamp"]) + 0.3)


def main() -> int:
    import cam_lock as L

    L.open_source("top")
    fr = L.frames("top", 6, gap=0.05)
    live = PT.median_markers([PT.detect(f) for f in fr], 0.5)
    now = {c: find_bowl(fr[-1], c) for _, c in BOWLS}
    print("live now: " + "  ".join(f"{c} ({b[0][0]:.0f},{b[0][1]:.0f})" if b else f"{c} -" for c, b in now.items()))
    targets = sys.argv[1:]
    if not targets:
        cfg = PT.cfg()
        targets = [PT.model_record(cfg[k])["dataset_id"] for k, _ in BOWLS]
    for t in targets:
        root = PT.STUDIO / "datasets" / t
        if not root.exists():
            rec = PT.model_record(t)
            root = PT.snapshot_dir(rec)
        print(f"\n== {t} ({root.name[:40]}) ==  ep: camera offset vs live (px) | bowl centres mapped into the live picture")
        rows = []
        for _, e in episodes(root).iterrows():
            f = frame(root, e)
            if f is None:
                print(f"ep {int(e['episode_index']):3d}: no frame")
                continue
            m = PT.detect(f)
            try:
                w = PT.fit_warp(m, live)  # this episode's picture -> live picture
            except SystemExit:
                print(f"ep {int(e['episode_index']):3d}: markers {sorted(m)} - can't place")
                continue
            H = np.array(w["H"])
            cells = []
            for _, c in BOWLS:
                b = find_bowl(f, c)
                if b is None:
                    cells.append(f"{c[:3]}    -    ")
                    continue
                p = cv2.perspectiveTransform(b[0][None, None], H)[0, 0]
                n = now.get(c)
                dist = f" d{np.hypot(*(p - n[0])):3.0f}" if n else ""
                cells.append(f"{c[:3]} ({p[0]:3.0f},{p[1]:3.0f}){dist}")
                rows.append((int(e["episode_index"]), c, p))
            print(f"ep {int(e['episode_index']):3d}: cam {w['moved_px']:4.0f} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
