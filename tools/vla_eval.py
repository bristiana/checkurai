#!/usr/bin/env python3
"""Offline check of a SmolVLA place model on its own demos, before it goes on the arm (CPU only: no arm, no cameras).

For episodes of both tasks (lilac bowl = good, orange bowl = defect) at several moments of each episode it feeds the
recorded observation (joint state, top + wrist frame, the episode's instruction) to the model's untouched OpenVINO
export (the *.orig.xml that tools/policy_tune.py keeps; the camera-drift warp is for the live camera, not the demos)
through physicalai's InferenceModel, as Studio does, and measures:
  1. open-loop fit: |predicted chunk - recorded future actions|, first 10 and 50 steps, per joint (degrees;
     gripper in Studio units 0..100), median / p90 over samples
  2. instruction use: the same observation with the other instruction. Where does the chunk end on shoulder_pan,
     relative to each bowl's release pan (measured in the demos: pan where the gripper last opens)? And how much do
     the actions change between the two instructions, compared with two runs of the same input (SmolVLA samples
     noise, so two runs differ a little anyway)?
  3. dataset facts: episodes per task, lengths, where the two tasks' pan trajectories split, release pan per bowl.
Results: results/vla_eval/<model>.json + .md.

    ~/physical-ai-studio/application/backend/.venv/bin/python tools/vla_eval.py smolvla-place-both-v7-a100
    ... tools/vla_eval.py <model> --episodes 8 --points 0,0.3,0.5,0.7 --device CPU   (~6-8 min on a busy CPU)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
DB = Path.home() / ".local/share/physicalai/data/physicalai.db"
OUT = ROOT / "results" / "vla_eval"
STAGE = Path("/tmp/vla_eval_export")
JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
LILAC = "put the brick from the spot into the lilac bowl"
ORANGE = "put the brick from the spot into the orange bowl"


def model_paths(name: str) -> tuple[Path, Path, str]:
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    r = c.execute("select m.path, d.path, m.policy from models m join datasets d on d.id = m.dataset_id where m.name = ?",
                  (name,)).fetchone()
    if r is None:
        raise SystemExit(f"no model {name!r} in Studio")
    return Path(r[0]), Path(r[1]), r[2]


def export_dir(model: Path, name: str, policy: str) -> tuple[Path, str]:
    """The model's OpenVINO export, with the untouched graph (policy_tune keeps it as <policy>.orig.xml/.bin).
    The export itself is never changed: a folder of symlinks in /tmp points <policy>.xml/.bin at the .orig files."""
    src = model / "exports" / "openvino"
    if not (src / f"{policy}.orig.xml").exists():
        return src, f"{policy}.xml (no .orig: not tuned)"
    dst = STAGE / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name.startswith(f"{policy}."):
            continue
        link = dst / f.name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(f)
    for ext in ("xml", "bin"):
        link = dst / f"{policy}.{ext}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src / f"{policy}.orig.{ext}")
    return dst, f"{policy}.orig.xml (untouched)"


def bowl(task: str) -> str:
    return "lilac" if "lilac" in task else "orange"


def dataset_facts(ds, act: np.ndarray, st: np.ndarray, task_idx: np.ndarray) -> tuple[dict, dict]:
    eps = ds.meta.episodes
    tasks = {int(i): t for t, i in ds.meta.tasks["task_index"].items()}
    out: dict = {"episodes": {}, "release_pan": {}, "release_at": {}, "length_s": {}, "odd": []}
    per: dict = {}
    for e in range(len(eps)):
        a, b = int(eps["dataset_from_index"][e]), int(eps["dataset_to_index"][e])
        task = tasks[int(task_idx[a])]
        g = act[a:b, 5]
        thr = (g.min() + g.max()) / 2
        ups = np.where((g[:-1] < thr) & (g[1:] >= thr))[0]
        rel = int(ups[-1]) + 1 if len(ups) else None
        moved = np.where(np.abs(act[a:b, :5] - act[a, :5]).max(axis=1) > 2.0)[0]  # first frame any joint moved > 2 deg
        per.setdefault(task, []).append({"ep": e, "a": a, "b": b, "len": b - a, "release": rel,
                                         "release_pan": float(st[a + rel, 0]) if rel is not None else None,
                                         "release_pose": st[a + rel, :4].tolist() if rel is not None else None,
                                         "idle_start": int(moved[0]) if len(moved) else b - a, "pan": st[a:b, 0]})
        if rel is None or g.max() - g.min() < 8:
            out["odd"].append(f"episode {e}: gripper never opens clearly (range {g.min():.1f}..{g.max():.1f})")
    for task, lst in per.items():
        key = bowl(task)
        lens = np.array([x["len"] for x in lst]) / ds.fps
        out["episodes"][key] = len(lst)
        out["length_s"][key] = {"median": round(float(np.median(lens)), 1), "min": round(float(lens.min()), 1),
                                "max": round(float(lens.max()), 1)}
        pans = [x["release_pan"] for x in lst if x["release_pan"] is not None]
        out["release_pan"][key] = {"median": round(float(np.median(pans)), 1),
                                   "p10": round(float(np.percentile(pans, 10)), 1),
                                   "p90": round(float(np.percentile(pans, 90)), 1)}
        out["release_at"][key] = round(float(np.median([x["release"] / x["len"] for x in lst if x["release"]])), 2)
        out.setdefault("release_pose_pan_lift_elbow_wflex", {})[key] = np.round(np.median(
            [x["release_pose"] for x in lst if x["release_pose"] is not None], axis=0), 1).tolist()
        idle = np.array([x["idle_start"] for x in lst]) / ds.fps
        out.setdefault("idle_at_start_s", {})[key] = {"median": round(float(np.median(idle)), 2),
                                                     "p90": round(float(np.percentile(idle, 90)), 2),
                                                     "max": round(float(idle.max()), 2)}
        med_len = np.median([y["len"] for y in lst])
        for x in lst:
            if x["len"] > 2.0 * med_len or x["len"] < 0.5 * med_len:
                out["odd"].append(f"episode {x['ep']} ({key}): {x['len'] / ds.fps:.1f} s vs median {med_len / ds.fps:.1f} s")
    # where the two tasks split: median pan per frame index, first frame where they differ by > 5 deg
    n = min(min(x["len"] for x in lst) for lst in per.values())
    med = {bowl(t): np.median(np.stack([x["pan"][:n] for x in lst]), axis=0) for t, lst in per.items()}
    diff = np.abs(med["lilac"] - med["orange"])
    split = int(np.argmax(diff > 5)) if (diff > 5).any() else None
    out["split_frame"] = split
    out["split_s"] = round(split / ds.fps, 1) if split is not None else None
    out["split_fraction_of_median_episode"] = (round(split / float(np.median([x["len"] for lst in per.values() for x in lst])), 2)
                                               if split is not None else None)
    return out, per


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--episodes", type=int, default=8, help="episodes per task")
    ap.add_argument("--points", default="0,0.3,0.5,0.7", help="moments of each episode (fraction of its length)")
    ap.add_argument("--noise-points", default="0,0.5", help="moments where the same input is run twice")
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()
    t_start = time.time()

    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from physicalai.inference import InferenceModel

    mpath, dpath, policy = model_paths(args.model)
    edir, graph = export_dir(mpath, args.model, policy)
    ds = LeRobotDataset("snapshot", root=dpath)
    df = pd.read_parquet(Path(ds.root) / "data")
    act = np.stack(df["action"].to_numpy()).astype(np.float64)
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    facts, per = dataset_facts(ds, act, st, df["task_index"].to_numpy())
    rel_pan = {k: v["median"] for k, v in facts["release_pan"].items()}
    rel_pose = {k: np.array(v) for k, v in facts["release_pose_pan_lift_elbow_wflex"].items()}
    t0 = time.time()
    model = InferenceModel(edir, backend="openvino", device=args.device)
    load_s = time.time() - t0
    points = [float(p) for p in args.points.split(",")]
    noise_points = {float(p) for p in args.noise_points.split(",")}

    def obs(i: int, task: str) -> dict:
        s = ds[i]
        o = {"state": s["observation.state"].numpy()[None].astype(np.float32), "task": [task]}
        for cam in ("top", "wrist"):
            img = s[f"observation.images.{cam}"].numpy()  # float CHW in [0, 1], RGB (as decoded for training)
            o[f"images.{cam}"] = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).transpose(1, 2, 0)[None]  # camera-like
        return o

    samples, inf_ms = [], []
    for task, lst in per.items():
        key = bowl(task)
        other = ORANGE if key == "lilac" else LILAC
        pick = [lst[int(round(j))] for j in np.linspace(0, len(lst) - 1, min(args.episodes, len(lst)))]
        for x in pick:
            for p in points:
                i = x["a"] + int(p * (x["len"] - 1))
                o = obs(i, task)
                t1 = time.time()
                ch = np.asarray(model.predict_action_chunk(o), np.float64)
                inf_ms.append((time.time() - t1) * 1000)
                sw = np.asarray(model.predict_action_chunk({**o, "task": [other]}), np.float64)
                rep = np.asarray(model.predict_action_chunk(o), np.float64) if p in noise_points else None
                fut = act[i:min(i + len(ch), x["b"])]
                k = len(fut)
                samples.append({"task": key, "ep": x["ep"], "p": p, "frame": i - x["a"],
                                "err10": np.abs(ch[:min(10, k)] - fut[:min(10, k)]).mean(axis=0).tolist(),
                                "err50": np.abs(ch[:k] - fut).mean(axis=0).tolist(),
                                "pan_now": float(o["state"][0, 0]), "pan_end": float(ch[k - 1, 0]),
                                "pan_end_swapped": float(sw[k - 1, 0]), "rec_pan_end": float(fut[-1, 0]),
                                "pose_end": ch[k - 1, :4].tolist(), "pose_end_swapped": sw[k - 1, :4].tolist(),
                                "rec_pose_end": fut[-1, :4].tolist(),
                                "d_instr": float(np.abs(ch[:, :5] - sw[:, :5]).mean()),
                                "d_noise": float(np.abs(ch[:, :5] - rep[:, :5]).mean()) if rep is not None else None})
                s = samples[-1]
                print(f"{key:6s} ep {x['ep']:2d} at {p:.1f}: err50 pan {s['err50'][0]:5.1f}  pan end {s['pan_end']:6.1f} "
                      f"(swapped {s['pan_end_swapped']:6.1f}, demo {s['rec_pan_end']:6.1f})  d_instr {s['d_instr']:.2f}"
                      + (f"  d_noise {s['d_noise']:.2f}" if s["d_noise"] is not None else ""), flush=True)

    def stats(key: str) -> dict:
        e = np.array([s[key] for s in samples])
        return {j: {"median": round(float(np.median(e[:, n])), 2), "p90": round(float(np.percentile(e[:, n], 90)), 2)}
                for n, j in enumerate(JOINTS)}

    def nearer(end, instr: str) -> bool:
        """Is the chunk-end pose (pan, lift, elbow, wrist flex) nearer the instructed bowl's release pose?
        The two bowls differ more in reach (lift / elbow) than in pan, so pan alone is not enough."""
        other = "orange" if instr == "lilac" else "lilac"
        e = np.asarray(end, np.float64)
        return float(np.linalg.norm(e - rel_pose[instr])) < float(np.linalg.norm(e - rel_pose[other]))

    def toward(s: dict) -> float:
        """How far the instruction swap moves the chunk-end pose along the lilac->orange release-pose axis
        (+ = towards the newly instructed bowl), in degrees."""
        axis = rel_pose["orange"] - rel_pose["lilac"]
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        d = (np.asarray(s["pose_end_swapped"]) - np.asarray(s["pose_end"])) @ axis
        return float(d if s["task"] == "lilac" else -d)

    sgn = np.sign(rel_pan["lilac"] - rel_pan["orange"])
    by_p = {}
    for p in points:
        sub = [s for s in samples if s["p"] == p]
        if not sub:
            continue
        # the swap moves the end pan towards the newly instructed bowl: lilac-instructed end minus orange-instructed end
        # has the sign of (lilac release pan - orange release pan)
        right_way = [np.sign((s["pan_end"] - s["pan_end_swapped"]) * (1 if s["task"] == "lilac" else -1)) == sgn
                     for s in sub]
        noise = [s["d_noise"] for s in sub if s["d_noise"] is not None]
        by_p[str(p)] = {
            "n": len(sub),
            "err50_pan_median": round(float(np.median([s["err50"][0] for s in sub])), 2),
            "nearer_instructed_true": round(float(np.mean([nearer(s["pose_end"], s["task"]) for s in sub])), 2),
            "nearer_instructed_swapped": round(float(np.mean(
                [nearer(s["pose_end_swapped"], "orange" if s["task"] == "lilac" else "lilac") for s in sub])), 2),
            "demo_nearer_own_bowl": round(float(np.mean([nearer(s["rec_pose_end"], s["task"]) for s in sub])), 2),
            "swap_moves_pose_toward_new_bowl_deg_median": round(float(np.median([toward(s) for s in sub])), 2),
            "swap_moves_pose_toward_new_bowl_frac": round(float(np.mean([toward(s) > 0 for s in sub])), 2),
            "swap_moves_pan_right_way": round(float(np.mean(right_way)), 2),
            "pan_shift_by_swap_median_deg": round(float(np.median([abs(s["pan_end"] - s["pan_end_swapped"]) for s in sub])), 2),
            "d_instr_mean": round(float(np.mean([s["d_instr"] for s in sub])), 3),
            "d_noise_mean": round(float(np.mean(noise)), 3) if noise else None,
        }
    res = {"model": args.model, "graph": graph, "device": args.device, "dataset": str(dpath), "facts": facts,
           "open_loop_err10": stats("err10"), "open_loop_err50": stats("err50"), "by_point": by_p,
           "load_s": round(load_s, 1), "infer_ms_median": round(float(np.median(inf_ms)), 0),
           "runtime_s": round(time.time() - t_start, 0), "samples": samples, "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.model}.json").write_text(json.dumps(res, indent=1))
    md = [f"# Offline check: {args.model}", "",
          f"{res['t']} (laptop clock) · {graph} · {args.device} · {len(samples)} samples · "
          f"infer {res['infer_ms_median']:.0f} ms median · run {res['runtime_s']:.0f} s", "",
          "## Open-loop fit: |predicted - recorded| per joint (deg; grip in Studio units 0..100)", "",
          "| | " + " | ".join(JOINTS) + " |", "|---|" + "---|" * len(JOINTS)]
    for k in ("open_loop_err10", "open_loop_err50"):
        md.append(f"| first {k[-2:]} steps, median / p90 | "
                  + " | ".join(f"{res[k][j]['median']} / {res[k][j]['p90']}" for j in JOINTS) + " |")
    md += ["", "## Instruction use", "",
           f"Release pose (pan, lift, elbow, wrist flex) from the demos: lilac {rel_pose['lilac'].tolist()}, "
           f"orange {rel_pose['orange'].tolist()}. Chunk end = predicted pose at the end of the chunk (same span as the "
           "demo's future). 'Swap toward new bowl' = how far swapping the instruction moves that pose along the "
           "lilac->orange axis, towards the newly instructed bowl.", "",
           "| moment | n | end nearer instructed bowl | same, instruction swapped | demo itself | swap toward new bowl "
           "(median deg / share > 0) | Δ actions, swap | Δ actions, same input |",
           "|---|---|---|---|---|---|---|---|"]
    for p, v in by_p.items():
        md.append(f"| {p} | {v['n']} | {v['nearer_instructed_true']} | {v['nearer_instructed_swapped']} | "
                  f"{v['demo_nearer_own_bowl']} | {v['swap_moves_pose_toward_new_bowl_deg_median']} / "
                  f"{v['swap_moves_pose_toward_new_bowl_frac']} | {v['d_instr_mean']} | {v['d_noise_mean']} |")
    md += ["", "## Dataset", "", "```", json.dumps(facts, indent=1), "```"]
    (OUT / f"{args.model}.md").write_text("\n".join(md) + "\n")
    print(f"wrote {OUT / (args.model + '.md')} in {res['runtime_s']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
