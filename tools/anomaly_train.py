#!/usr/bin/env python3
"""Train, export and evaluate the brick anomaly models (Anomalib -> OpenVINO), plus the empty-spot check.

Data: data/inspection/{good,defect,empty,good_other}/*.jpg from tools/capture_inspection.py (1920x1080).

    python tools/anomaly_train.py roi        # fixed square ROI -> config/inspection_roi.json
    python tools/anomaly_train.py prepare    # ROI crops + per-brick split -> data/inspection_crops/
    python tools/anomaly_train.py prepare-brick --side 0    # tight brick crops -> data/inspection_crops_bricktight/
    python tools/anomaly_train.py prepare-brick --side -1   # fixed-side brick crops (side fits every brick)
    python tools/anomaly_train.py train --model patchcore --crops bricktight --size 384
    python tools/anomaly_train.py train --model patchcore --size 384      # whole fixed ROI (v1)
    python tools/anomaly_train.py train --model patchcore --size 384 --all-faces
    python tools/anomaly_train.py train --model efficientad --size 256 --steps 2000
    python tools/anomaly_train.py eval --name patchcore_384     # scores, threshold + band, latency CPU/GPU/NPU
    python tools/anomaly_train.py empty      # empty-spot check reference + accuracy
    python tools/anomaly_train.py report     # results/anomaly/summary.json + anomaly.png

Split (written to results/anomaly/split.json): the photos of each brick are taken in order, each one a new
placement and angle. Per brick, every 10 photos -> 7 train, 3 held out (alternately calibration / test),
so every brick type is in training (all of them appear in deployment) and the held-out photos are
placements the model has not seen. All defect photos are test only. The threshold is set on the
calibration good photos only; the test good + defect photos are only used for the reported numbers.

Threshold: low = highest calibration good score, high = low + (low - median calibration good score).
score < low -> good; score >= high -> defect; in between -> re-inspect, and if it is still >= low, defect.
Policy: an uncertain brick goes to the defect bowl, so the reported recall / false alarms count every
score >= low as defect. Eval also swaps calibration and test (2 folds) so all 27 held-out goods count.
Caveat: all good photos come from one capture session and all defect photos from a later one, so part of
the gap can be scene drift; eval also checks that the anomaly-map peak lies on the brick.

Brick crops: the brick is the largest blob that differs from the empty-spot reference inside the ROI;
"tight" = its box plus a margin (each brick fills the model input), a fixed side keeps one scale. The
exported model carries the crop settings and the reference, so AnomalyScorer still takes the ROI crop.
The photos are single shots about 3 s apart with the brick moved each time (median 64 px), so there are
no bursts to measure a median-of-N effect on.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspection import (ROI_PATH, VIEWS_PATH, AnomalyScorer, EmptyCheck, MultiViewScorer, brick_window, crop,  # noqa: E402
                        load_roi, load_views, view_crops)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "inspection"
CROPS = ROOT / "data" / "inspection_crops"
MODELS = ROOT / "models" / "anomaly"
RESULTS = ROOT / "results" / "anomaly"
IMAGENETTE = ROOT / "data" / "imagenette"

# capture sessions (HHMMSS of the file name, inclusive)
GOOD_BRICKS = [
    ("red", "155851", "160031"), ("red", "160939", "161007"),
    ("green_small", "160050", "160100"), ("green_small", "161012", "161039"),
    ("lightblue_2x4", "160236", "160444"), ("green_arch", "160452", "160523"),
    ("pink", "160528", "160609"), ("blue_2x2", "160615", "160823"),
]
DEFECT_TYPES = [("tape", "161702", "161841"), ("marker_line", "161922", "162026"), ("marker_dot", "162053", "162210")]
SEARCH = (680, 1320, 130, 1080)  # x0, x1, y0, y1: between the two bowls, below the robot base
DEVICES = ("CPU", "GPU", "NPU")


def hhmmss(p: Path) -> str:
    return p.stem.split("-")[1]


def lookup(t: str, table) -> str | None:
    return next((name for name, a, b in table if a <= t <= b), None)


def photos(cls: str) -> list[Path]:
    return sorted((DATA / cls).glob("*.jpg"))


# ---------------------------------------------------------------- ROI
def cmd_roi(args) -> None:
    s = 4
    x0, x1, y0, y1 = SEARCH
    small = lambda p: cv2.GaussianBlur(cv2.resize(cv2.imread(str(p)), (1920 // s, 1080 // s), interpolation=cv2.INTER_AREA), (5, 5), 0)
    med = np.median(np.stack([small(p) for p in photos("empty")]), axis=0).astype(np.uint8)
    boxes = []
    for p in photos("good") + photos("defect"):
        d = cv2.absdiff(small(p), med).max(axis=2)[y0 // s:y1 // s, x0 // s:x1 // s]
        m = cv2.morphologyEx((d > 40).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            print(f"  no brick found in {p.name} (empty photo in the wrong folder?)")
            continue
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh, area = st[i]
        if area * s * s < 5000:
            print(f"  only a small change in {p.name} ({area * s * s} px), skipped")
            continue
        boxes.append((bx * s + x0, by * s + y0, (bx + bw) * s + x0, (by + bh) * s + y0))
    b = np.array(boxes)
    ux0, uy0, ux1, uy1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = int(max(ux1 - ux0, uy1 - uy0) + 2 * args.margin)
    cx, cy = (ux0 + ux1) // 2, (uy0 + uy1) // 2
    x, y = int(np.clip(cx - side // 2, 0, 1920 - side)), int(np.clip(cy - side // 2, 0, 1080 - side))
    roi = {"x": x, "y": y, "size": side, "frame": [1920, 1080],
           "made_from": f"union of {len(boxes)} brick boxes (good + defect) + {args.margin}px margin",
           "brick_box_union": [int(ux0), int(uy0), int(ux1), int(uy1)]}
    ROI_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROI_PATH.write_text(json.dumps(roi, indent=2) + "\n")
    print(f"ROI x={x} y={y} size={side} -> {ROI_PATH}")


# ---------------------------------------------------------------- crops + split
def cmd_prepare(args) -> None:
    roi = load_roi()
    if CROPS.exists():
        shutil.rmtree(CROPS)
    split = []
    by_brick: dict[str, list[Path]] = {}
    for p in photos("good"):
        brick = lookup(hhmmss(p), GOOD_BRICKS)
        if brick is None:
            sys.exit(f"good photo {p.name} is not in any brick session; update GOOD_BRICKS")
        by_brick.setdefault(brick, []).append(p)
    for brick, ps in by_brick.items():
        held = 0
        for i, p in enumerate(ps):
            if i % 10 in (2, 5, 8):
                part = "calib_good" if held % 2 == 0 else "test_good"
                held += 1
            else:
                part = "train_good"
            split.append({"file": p.name, "class": "good", "brick": brick, "part": part})
    for p in photos("defect"):
        split.append({"file": p.name, "class": "defect", "type": lookup(hhmmss(p), DEFECT_TYPES) or "other", "part": "defect"})
    for p in photos("good_other"):
        split.append({"file": p.name, "class": "good_other", "part": "good_other"})
    for p in photos("empty"):
        split.append({"file": p.name, "class": "empty", "part": "empty"})
    for row in split:
        out = CROPS / row["part"] / row["file"]
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), crop(cv2.imread(str(DATA / row["class"] / row["file"])), roi), [cv2.IMWRITE_JPEG_QUALITY, 95])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "split.json").write_text(json.dumps(split, indent=1))
    counts = {}
    for row in split:
        counts[row["part"]] = counts.get(row["part"], 0) + 1
    print("crops:", counts, "| bricks:", {k: len(v) for k, v in by_brick.items()})


def crop_tag(side: int) -> str:
    return f"brick{side}" if side else "bricktight"


def cmd_prepare_brick(args) -> None:
    """Brick-centred crops of the ROI crops (same split), the same way AnomalyScorer makes them at run time."""
    split = json.loads((RESULTS / "split.json").read_text())
    empty = EmptyCheck.load(MODELS / "empty_ref.png")
    boxes = {}
    for r in split:
        if r["part"] == "empty":
            continue
        boxes[r["file"]] = empty.brick_box(cv2.imread(str(CROPS / r["part"] / r["file"])))
    found = [b for b in boxes.values() if b]
    w = np.array([b[2] - b[0] for b in found])
    h = np.array([b[3] - b[1] for b in found])
    longest = np.maximum(w, h)
    print(f"brick found in {len(found)}/{len(boxes)} photos; longest box edge px: median {np.median(longest):.0f}, "
          f"p95 {np.percentile(longest, 95):.0f}, max {longest.max()}")
    side = args.side
    if side < 0:  # auto: every brick fits with the margin
        side = int(np.percentile(longest, 99) + 2 * args.margin)
    out_dir = ROOT / "data" / f"inspection_crops_{crop_tag(side)}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for r in split:
        if r["part"] == "empty":
            continue
        roi_img = cv2.imread(str(CROPS / r["part"] / r["file"]))
        x, y, s = brick_window(roi_img.shape[0], boxes[r["file"]], side, args.margin)
        p = out_dir / r["part"] / r["file"]
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), roi_img[y:y + s, x:x + s], [cv2.IMWRITE_JPEG_QUALITY, 95])
    cfg = {"mode": "brick", "side": side or None, "margin": args.margin, "empty_ref": "empty_ref.png"}
    (out_dir / "crop.json").write_text(json.dumps(cfg))
    # consecutive good photos of one brick: how far did the brick move between shots?
    goods = sorted((r for r in split if r["class"] == "good"), key=lambda r: r["file"])
    shifts = []
    for a, b in zip(goods, goods[1:], strict=False):
        if a["brick"] == b["brick"] and boxes[a["file"]] and boxes[b["file"]]:
            ca = np.add(boxes[a["file"]][:2], boxes[a["file"]][2:]) / 2
            cb = np.add(boxes[b["file"]][:2], boxes[b["file"]][2:]) / 2
            shifts.append(float(np.hypot(*(ca - cb))))
    print(f"crop side {side or 'tight'} px (ROI is {roi_img.shape[0]}) -> {out_dir}")
    print(f"brick centre shift between consecutive good photos: median {np.median(shifts):.0f} px, "
          f"min {np.min(shifts):.0f}, <10 px in {int((np.array(shifts) < 10).sum())}/{len(shifts)}")


# ---------------------------------------------------------------- training
def engine_for(device: str, **kw):
    from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator

    if device == "xpu":
        return Engine(accelerator=XPUAccelerator(), strategy=SingleXPUStrategy(), logger=False, **kw)
    return Engine(accelerator="cpu", logger=False, **kw)


def cmd_train(args) -> None:
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.models import EfficientAd, Patchcore

    root, crop_cfg = CROPS, None
    if args.crops:  # brick-centred crops from prepare-brick
        root = ROOT / "data" / f"inspection_crops_{args.crops}"
        crop_cfg = json.loads((root / "crop.json").read_text())
    name = args.name or (f"{args.model}_" + (f"{args.crops}_" if args.crops else "") + f"{args.size}" + ("_allfaces" if args.all_faces else ""))
    out = MODELS / name
    if out.exists():
        shutil.rmtree(out)
    normal = ["train_good"] + (["good_other"] if args.all_faces else [])
    size = (args.size, args.size)
    t0 = time.time()
    if args.model == "patchcore":
        dm = Folder(name=name, root=root, normal_dir=normal, abnormal_dir="defect", normal_test_dir="test_good",
                    train_batch_size=8, eval_batch_size=8, num_workers=4, val_split_mode="none", seed=0)
        model = Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), coreset_sampling_ratio=args.coreset,
                          pre_processor=Patchcore.configure_pre_processor(image_size=size),
                          post_processor=False, evaluator=False, visualizer=False)
        engine = engine_for(args.device, default_root_dir=str(out / "work"), max_epochs=1, limit_val_batches=0)
        engine.fit(model=model, datamodule=dm)
    else:
        dm = Folder(name=name, root=root, normal_dir=normal, abnormal_dir="defect", normal_test_dir="test_good",
                    train_batch_size=1, eval_batch_size=8, num_workers=4, val_split_mode="from_train",
                    val_split_ratio=0.15, seed=0)
        model = EfficientAd(imagenet_dir=IMAGENETTE, model_size="small",
                            pre_processor=EfficientAd.configure_pre_processor(image_size=size),
                            post_processor=False, evaluator=False, visualizer=False)
        engine = engine_for(args.device, default_root_dir=str(out / "work"), max_steps=args.steps,
                            check_val_every_n_epoch=1000)
        engine.fit(model=model, datamodule=dm)
        engine.validate(model=model, datamodule=dm)  # map-normalisation quantiles from the final weights
    train_s = time.time() - t0
    engine.export(model=model, export_type=ExportType.OPENVINO, export_root=out, input_size=size)
    xml = next(out.rglob("*.xml"))
    meta = {"name": name, "model": args.model, "image_size": args.size, "normal_dirs": normal,
            "train_seconds": round(train_s, 1), "device": args.device, "xml": str(xml.relative_to(out))}
    if args.model == "efficientad":
        meta["steps"] = args.steps
    else:
        meta["coreset"] = args.coreset
    if crop_cfg:  # the scorer finds the brick in the ROI itself, with this empty-spot reference
        meta["crop"] = crop_cfg
        refs = ([VIEW_REFS / r for r in crop_cfg["empty_refs"].values()] if crop_cfg.get("mode") in ("views", "view")
                else [MODELS / "empty_ref.png"])
        for ref in refs:
            for ext in (".png", ".json"):
                shutil.copy(ref.with_suffix(ext), xml.parent / ref.with_suffix(ext).name)
    (xml.parent / "checkurai_meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(out / "work", ignore_errors=True)
    print(f"trained {name} in {train_s:.0f}s -> {xml}")


# ---------------------------------------------------------------- evaluation
def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Probability that a random defect scores above a random good (ties count half)."""
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def latency(xml: Path, size: int, device: str, n: int = 30) -> dict:
    try:
        t = time.time()
        sc = AnomalyScorer(xml, device=device, image_size=size, fallback=None)
        compile_s = time.time() - t
    except Exception as e:  # noqa: BLE001 - report any compile failure
        return {"ok": False, "error": str(e).splitlines()[0][:200]}
    img = np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8)
    for _ in range(5):
        sc(img)
    ms = [sc(img)[2] for _ in range(n)]
    return {"ok": True, "median_ms": round(float(np.median(ms)), 2), "p90_ms": round(float(np.percentile(ms, 90)), 2),
            "compile_s": round(compile_s, 1)}


def cmd_eval(args) -> None:
    xml = next((MODELS / args.name).rglob("*.xml"))
    meta = json.loads((xml.parent / "checkurai_meta.json").read_text())
    split = json.loads((RESULTS / "split.json").read_text())
    sc = AnomalyScorer(xml, device="CPU")  # the scorer gets the ROI crop, exactly as in the cell program
    S = sc.size
    small = lambda im: cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32)
    empty_med = np.median(np.stack([cv2.imread(str(CROPS / "empty" / r["file"])) for r in split if r["part"] == "empty"]),
                          axis=0).astype(np.uint8)
    rows = []
    for r in split:
        if r["part"] in ("train_good", "empty"):
            continue
        im = cv2.imread(str(CROPS / r["part"] / r["file"]))
        s, amap, _ = sc(im)
        x, y, w = sc.last_window
        on_brick = None
        if amap is not None:  # does the anomaly peak sit on the brick (not on the background)?
            diff = np.abs(small(im[y:y + w, x:x + w]) - small(empty_med[y:y + w, x:x + w])).max(axis=2)
            brick = cv2.dilate((cv2.GaussianBlur(diff, (9, 9), 0) > 40).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            amap = cv2.resize(np.asarray(amap, np.float32), (S, S))
            on_brick = bool(brick[np.unravel_index(np.argmax(amap), amap.shape)])
        rows.append({**r, "score": s, "peak_on_brick": on_brick, "window": [x, y, w]})
    get = lambda part, key=None, val=None: np.array([r["score"] for r in rows if r["part"] == part and (key is None or r.get(key) == val)])
    calib, test, defect = get("calib_good"), get("test_good"), get("defect")
    low = float(calib.max())
    high = float(low + (low - np.median(calib)))
    heldout = np.concatenate([calib, test])
    tp = int((defect >= low).sum())
    fn = int((defect < low).sum())
    fp = int((test >= low).sum())
    tn = int((test < low).sum())
    res = {
        "name": args.name, **{k: meta[k] for k in ("model", "image_size", "normal_dirs", "train_seconds")},
        "counts": {"calib_good": len(calib), "test_good": len(test), "defect": len(defect)},
        "auroc_heldout_good_vs_defect": round(auroc(heldout, defect), 4),
        "auroc_test_good_vs_defect": round(auroc(test, defect), 4),
        "threshold": {"low": low, "high": high,
                      "rule": "score < low: good; low..high: re-inspect (median of frames), still >= low: defect; >= high: defect",
                      "low_from": "max score of the calibration good photos", "high_from": "low + (low - median calibration good score)"},
        "confusion_at_low": {"defect_as_defect": tp, "defect_as_good": fn, "good_as_defect": fp, "good_as_good": tn},
        "in_band": {"test_good": int(((test >= low) & (test < high)).sum()), "defect": int(((defect >= low) & (defect < high)).sum())},
        "recall_per_type": {},
        "good_other_flagged": None,
        "scores": {"calib_good": calib.round(4).tolist(), "test_good": test.round(4).tolist(),
                   "defect": {t: get("defect", "type", t).round(4).tolist() for t, _, _ in DEFECT_TYPES}},
    }
    for t, _, _ in DEFECT_TYPES:
        d = get("defect", "type", t)
        res["recall_per_type"][t] = {"n": int(len(d)), "recall": round(float((d >= low).mean()), 3) if len(d) else None,
                                     "auroc": round(auroc(heldout, d), 4)}
    # policy: uncertain (inside the band) goes to the defect bowl, so a call is "defect" at score >= low.
    # Two folds (threshold from calibration, tested on test; then the other way round) so every held-out good
    # photo counts once in the false-alarm rate.
    low2 = float(test.max())
    res["policy_uncertain_is_defect"] = {
        "false_alarm_rate_test": f"{fp}/{len(test)}",
        "false_alarm_rate_2fold": f"{fp + int((calib >= low2).sum())}/{len(test) + len(calib)}",
        "recall_2fold": {t: round(float(((get('defect', 'type', t) >= low).mean() + (get('defect', 'type', t) >= low2).mean()) / 2), 3)
                         for t, _, _ in DEFECT_TYPES},
        "recall_all_2fold": round(float(((defect >= low).mean() + (defect >= low2).mean()) / 2), 3),
    }
    if sc.crop_cfg:
        wins = np.array([r["window"][2] for r in rows if r["part"] != "good_other"])
        roi_side = load_roi()["size"]
        res["crop"] = {**sc.crop_cfg, "window_side_median": int(np.median(wins)),
                       "no_brick_found_whole_roi": int((wins == roi_side).sum())}
    peaks = {}
    for r in rows:
        if r["peak_on_brick"] is not None and r["part"] != "good_other":
            g = peaks.setdefault(r.get("type", "good"), [0, 0])
            g[0] += r["peak_on_brick"]
            g[1] += 1
    res["anomaly_peak_on_brick"] = {k: f"{a}/{n}" for k, (a, n) in peaks.items()}
    other = get("good_other")
    if len(other) and "good_other" not in meta["normal_dirs"]:
        res["good_other_flagged"] = {"n": int(len(other)), "flagged": int((other >= low).sum())}
    out = RESULTS / f"{args.name}.json"
    if not args.no_latency:
        res["latency"] = {dev: latency(xml, meta["image_size"], dev) for dev in DEVICES}
    elif out.exists() and "latency" in json.loads(out.read_text()):  # keep the earlier benchmark
        res["latency"] = json.loads(out.read_text())["latency"]
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "scores"}, indent=1))


# ---------------------------------------------------------------- empty-spot check
def cmd_empty(args) -> None:
    split = json.loads((RESULTS / "split.json").read_text())
    img = lambda r: cv2.imread(str(CROPS / r["part"] / r["file"]))
    empties = [r for r in split if r["part"] == "empty"]
    ref_rows, eval_rows = empties[0::2], empties[1::2]
    chk = EmptyCheck.from_crops([img(r) for r in ref_rows], pixel_thr=args.pixel_thr)
    frac_empty = np.array([chk.changed_fraction(img(r)) for r in eval_rows])
    bricks = [r for r in split if r["part"] in ("train_good", "calib_good", "test_good", "defect", "good_other")]
    frac_brick = np.array([chk.changed_fraction(img(r)) for r in bricks])
    thr = float(np.sqrt(max(frac_empty.max(), 1e-4) * frac_brick.min())) if frac_brick.min() > frac_empty.max() else args.frac_thr
    acc_empty = float((frac_empty <= thr).mean())
    acc_brick = float((frac_brick > thr).mean())
    final = EmptyCheck.from_crops([img(r) for r in empties], pixel_thr=args.pixel_thr, frac_thr=thr)
    MODELS.mkdir(parents=True, exist_ok=True)
    final.save(MODELS / "empty_ref.png")
    res = {"pixel_thr": args.pixel_thr, "frac_thr": round(thr, 5),
           "eval": {"empty_n": len(eval_rows), "empty_correct": round(acc_empty, 4),
                    "brick_n": len(bricks), "brick_correct": round(acc_brick, 4)},
           "empty_frac_max": round(float(frac_empty.max()), 5), "brick_frac_min": round(float(frac_brick.min()), 5),
           "method": "reference = median of half the empty crops; the other half + every brick photo are the test"}
    (RESULTS / "empty_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- report
def cmd_report(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))
            if p.stem not in ("split", "summary", "empty_check") and not p.stem.startswith(("split_views", "views_empty_check"))]
    empty = json.loads((RESULTS / "empty_check.json").read_text()) if (RESULTS / "empty_check.json").exists() else None
    summary = {"models": {r["name"]: {k: v for k, v in r.items() if k != "scores"} for r in runs}, "empty_check": empty}
    table = {}
    for r in runs:
        pol = r.get("policy_uncertain_is_defect", {})
        lat = r.get("latency", {})
        table[r["name"]] = {
            "auroc": r["auroc_heldout_good_vs_defect"],
            "recall_at_low": {t: v["recall"] for t, v in r["recall_per_type"].items()},
            "false_alarms_test": pol.get("false_alarm_rate_test"),
            "false_alarms_2fold": pol.get("false_alarm_rate_2fold"),
            "recall_2fold": pol.get("recall_2fold"),
            "ms_cpu_gpu_npu": [lat.get(d, {}).get("median_ms") for d in DEVICES],
        }
        print(f"{r['name']:28s} AUROC {r['auroc_heldout_good_vs_defect']:.3f} recall {table[r['name']]['recall_at_low']} "
              f"FA {pol.get('false_alarm_rate_test')} / 2-fold {pol.get('false_alarm_rate_2fold')} ms {table[r['name']]['ms_cpu_gpu_npu']}")
    summary["table"] = table
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    by_name = {r["name"]: r for r in runs}
    show = [by_name[n] for n in args.plot if n in by_name] or runs[:2]
    fig, axes = plt.subplots(len(show), 2, figsize=(12, 4.2 * len(show)), squeeze=False)
    for ax_h, ax_c, r in [(a[0], a[1], r) for a, r in zip(axes, show, strict=False)]:
        s = r["scores"]
        everything = s["calib_good"] + s["test_good"] + sum(s["defect"].values(), [])
        bins = np.linspace(min(everything), max(everything), 40)
        ax_h.hist(s["calib_good"] + s["test_good"], bins=bins, alpha=0.7, label="good (held out)", color="#2f6fd8")
        for (t, v), c in zip(s["defect"].items(), ["#d6452f", "#f0a030", "#8e44ad", "#2a9d5c", "#c0398f", "#6b5b3e", "#444444"], strict=False):
            ax_h.hist(v, bins=bins, alpha=0.6, label=f"defect: {t}", color=c)
        ax_h.axvline(r["threshold"]["low"], color="k", ls="--", lw=1, label="threshold")
        ax_h.axvspan(r["threshold"]["low"], r["threshold"]["high"], color="0.5", alpha=0.15, label="re-inspect band")
        ax_h.set_title(f"{r['name']}: anomaly scores (AUROC {r['auroc_heldout_good_vs_defect']:.3f})")
        ax_h.set_xlabel("raw score")
        ax_h.legend(fontsize=8)
        c = r["confusion_at_low"]
        m = np.array([[c["good_as_good"], c["good_as_defect"]], [c["defect_as_good"], c["defect_as_defect"]]])
        ax_c.imshow(m, cmap="Blues")
        for (i, j), v in np.ndenumerate(m):
            ax_c.text(j, i, str(v), ha="center", va="center", fontsize=16, color="k" if v < m.max() / 2 else "w")
        ax_c.set_xticks([0, 1], ["said good", "said defect"])
        ax_c.set_yticks([0, 1], ["is good", "is defect"])
        ax_c.set_title("test set at the threshold")
    fig.tight_layout()
    fig.savefig(RESULTS / "anomaly.png", dpi=110)
    print(f"wrote {RESULTS / 'summary.json'} and {RESULTS / 'anomaly.png'}")


# ---------------------------------------------------------------- mirror booth: direct view + two mirrors
# Photos from tools/capture_session.sh (data/inspection/{empty,good,defect}). Which brick / defect kind a photo
# shows comes from the session log (start/end time of every step) -> config/inspection_sessions.json.
SESSIONS_PATH = ROOT / "config" / "inspection_sessions.json"
VIEW_SEARCH = {"direct": (560, 1540, 520, 1080), "left": (200, 950, 20, 620), "right": (950, 1720, 20, 600)}  # x0 x1 y0 y1
VIEW_MIN_AREA = {"direct": 8000, "left": 3000, "right": 3000}  # full-res px; smaller changes are not a brick
VIEW_REFS = MODELS / "views"
VCROPS = ROOT / "data" / "inspection_crops_views"
SPLIT_VIEWS = RESULTS / "split_views.json"
SESSION_LOG = ROOT / "logs" / "capture_session.log"
VIEWS_CHECK = RESULTS / "views_empty_check.json"


def use_set(tag: str) -> None:
    """--set TAG: a separate mirror-booth photo set (e.g. after the booth was moved or relit). Every booth path gets
    the tag, so the original set, its view rects, references, split and models stay as they are:
    data/inspection_TAG/, config/inspection_{views,sessions}_TAG.json, models/anomaly/views_TAG/,
    data/inspection_crops_views_TAG/ (train --crops views_TAG), results/anomaly/split_views_TAG.json,
    logs/capture_session_TAG.log."""
    global DATA, VIEWS_PATH, VIEW_REFS, VCROPS, SESSIONS_PATH, SPLIT_VIEWS, SESSION_LOG, VIEWS_CHECK
    DATA = ROOT / "data" / f"inspection_{tag}"
    VIEWS_PATH = ROOT / "config" / f"inspection_views_{tag}.json"
    SESSIONS_PATH = ROOT / "config" / f"inspection_sessions_{tag}.json"
    VIEW_REFS = MODELS / f"views_{tag}"
    VCROPS = ROOT / "data" / f"inspection_crops_views_{tag}"
    SPLIT_VIEWS = RESULTS / f"split_views_{tag}.json"
    SESSION_LOG = ROOT / "logs" / f"capture_session_{tag}.log"
    VIEWS_CHECK = RESULTS / f"views_empty_check_{tag}.json"
STEP_TYPES = {"defect_tape": "tape", "defect_marker_line": "marker_line", "marker_line2": "marker_line",
              "defect_marker_dot": "marker_dot", "defect_other": "other", "marker_bigdot": "marker_bigdot",
              "marker_smalldot": "marker_smalldot"}


def small4(p: Path) -> np.ndarray:
    return cv2.GaussianBlur(cv2.resize(cv2.imread(str(p)), (480, 270), interpolation=cv2.INTER_AREA), (5, 5), 0).astype(np.float32)


def blob_box(diff: np.ndarray, region: tuple[int, int, int, int], thr: float = 40) -> tuple[tuple[int, int, int, int], int] | None:
    """Largest changed blob inside a full-frame region of a 1/4-scale diff -> (full-frame box, area px)."""
    x0, x1, y0, y1 = region
    m = cv2.morphologyEx((diff[y0 // 4:y1 // 4, x0 // 4:x1 // 4] > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh, area = (int(v) for v in st[i])
    return (bx * 4 + x0, by * 4 + y0, (bx + bw) * 4 + x0, (by + bh) * 4 + y0), area * 16


def empty_median() -> np.ndarray:
    return np.median(np.stack([small4(p) for p in photos("empty")]), axis=0)


def cmd_sessions(args) -> None:
    """logs/capture_session.log -> config/inspection_sessions.json: [name, HHMMSS start, HHMMSS end] per step."""
    good, defect, started = [], [], {}
    for line in SESSION_LOG.read_text().splitlines():
        f = line.split()
        if len(f) < 3 or f[1] not in ("start", "end"):
            continue
        if f[1] == "start":
            started[f[2]] = f[0]
        elif f[2] in started:
            a = started.pop(f[2])
            if f[2].startswith("good"):
                good.append([f"brick{f[2][4:]}", a, f[0]])
            elif f[2] in STEP_TYPES:
                defect.append([STEP_TYPES[f[2]], a, f[0]])
    SESSIONS_PATH.write_text(json.dumps({"good": good, "defect": defect}, indent=1) + "\n")
    print(f"{len(good)} good bricks, {len(defect)} defect steps -> {SESSIONS_PATH}")


def cmd_screen(args) -> None:
    """Move photos without a brick (saved while swapping bricks) and near-duplicates to data/inspection/_junk/."""
    med = empty_median()
    prev, moved = None, []
    for cls in ("good", "defect"):
        for p in photos(cls):
            im = small4(p)
            diff = np.abs(im - med).max(axis=2)
            found = blob_box(diff, VIEW_SEARCH["direct"])
            dup = prev is not None and float((np.abs(im - prev).max(axis=2) > 25).mean()) < 0.002
            prev = im
            why = "no brick" if not found or found[1] < VIEW_MIN_AREA["direct"] else "duplicate" if dup else None
            if why:
                moved.append(p)
                print(f"  {cls}/{p.name}: {why}")
                if not args.dry_run:
                    (DATA / "_junk" / cls).mkdir(parents=True, exist_ok=True)
                    p.rename(DATA / "_junk" / cls / p.name)
    print(f"{len(moved)} photos {'would be ' if args.dry_run else ''}moved to {DATA / '_junk'}")


def cmd_views(args) -> None:
    """View rectangles (union of the brick boxes per view + margin) and one empty-spot reference per view."""
    med = empty_median()
    boxes: dict[str, list] = {n: [] for n in VIEW_SEARCH}
    for p in photos("good") + photos("defect"):
        diff = np.abs(small4(p) - med).max(axis=2)
        for n, region in VIEW_SEARCH.items():
            found = blob_box(diff, region)
            if found and found[1] >= VIEW_MIN_AREA[n]:
                boxes[n].append(found[0])
            else:
                print(f"  no brick in the {n} view of {p.parent.name}/{p.name}")
    rects = {}
    for n, bs in boxes.items():
        b = np.array(bs)
        x0, y0 = max(int(b[:, 0].min()) - args.margin, 0), max(int(b[:, 1].min()) - args.margin, 0)
        x1, y1 = min(int(b[:, 2].max()) + args.margin, 1920), min(int(b[:, 3].max()) + args.margin, 1080)
        rects[n] = [x0, y0, x1, y1]
        print(f"{n}: {len(bs)} boxes -> view {rects[n]} ({x1 - x0}x{y1 - y0})")
    views = {"frame": [1920, 1080], "views": rects, "made_from": f"union of the brick boxes per view + {args.margin}px"}
    VIEWS_PATH.write_text(json.dumps(views, indent=1) + "\n")
    # empty references: reference from half the empty photos, tested on the other half + every brick photo
    empties = [view_crops(cv2.imread(str(p)), views) for p in photos("empty")]
    bricks = [view_crops(cv2.imread(str(p)), views)["direct"] for p in photos("good") + photos("defect")]
    VIEW_REFS.mkdir(parents=True, exist_ok=True)
    res = {}
    for n in rects:
        chk = EmptyCheck.from_crops([e[n] for e in empties[0::2]], pixel_thr=args.pixel_thr)
        fe = np.array([chk.changed_fraction(e[n]) for e in empties[1::2]])
        thr = 0.02
        if n == "direct":  # the spot check (is a brick there?) uses the direct view only
            fb = np.array([chk.changed_fraction(c) for c in bricks])
            thr = float(np.sqrt(max(fe.max(), 1e-4) * fb.min())) if fb.min() > fe.max() else 0.02
            res = {"pixel_thr": args.pixel_thr, "frac_thr": round(thr, 5), "empty_frac_max": round(float(fe.max()), 5),
                   "brick_frac_min": round(float(fb.min()), 5), "empty_correct": float((fe <= thr).mean()),
                   "brick_correct": float((fb > thr).mean()), "n_empty_eval": len(fe), "n_brick": len(fb)}
        EmptyCheck.from_crops([e[n] for e in empties], pixel_thr=args.pixel_thr, frac_thr=thr).save(VIEW_REFS / f"{n}_empty.png")
    RESULTS.mkdir(parents=True, exist_ok=True)
    VIEWS_CHECK.write_text(json.dumps(res, indent=1))
    print(f"views -> {VIEWS_PATH}; empty refs -> {VIEW_REFS}; spot check: {res}")


def cmd_prepare_views(args) -> None:
    """Per-brick split (7 train / 3 held out per 10 photos) and one brick crop per view -> data/inspection_crops_views/.

    --out TAG: another crop set (e.g. a smaller margin) -> data/inspection_crops_views_TAG/, made from the existing
    results/anomaly/split_views.json (same split, not rewritten), no per-view folders; train it with --crops views_TAG.
    """
    if args.out:
        return prepare_views_tagged(args)
    ses = json.loads(SESSIONS_PATH.read_text())
    views = load_views(VIEWS_PATH)
    refs = {n: EmptyCheck.load(VIEW_REFS / f"{n}_empty.png") for n in views["views"]}
    split, by_brick = [], {}
    for p in photos("good"):
        brick = lookup(hhmmss(p), ses["good"])
        if brick is None:
            print(f"  good/{p.name} is in no brick step of {SESSIONS_PATH.name}, skipped")
            continue
        by_brick.setdefault(brick, []).append(p)
    held = 0  # alternates calibration / test across bricks, so both get about half of the held-out photos
    held_out = tuple(int(v) for v in args.held_out.split(","))  # photo index (mod 10) within a brick
    for brick, ps in by_brick.items():
        for i, p in enumerate(ps):
            part = "train_good"
            if i % 10 in held_out:
                part = "calib_good" if held % 2 == 0 else "test_good"
                held += 1
            split.append({"file": p.name, "class": "good", "brick": brick, "part": part})
    for p in photos("defect"):
        split.append({"file": p.name, "class": "defect", "type": lookup(hhmmss(p), ses["defect"]) or "other", "part": "defect"})
    for d in [VCROPS] + [Path(f"{VCROPS}_{n}") for n in views["views"]]:
        if d.exists():
            shutil.rmtree(d)
    missing = {n: 0 for n in views["views"]}
    for r in split:
        for n, v in view_crops(cv2.imread(str(DATA / r["class"] / r["file"])), views).items():
            box = refs[n].brick_box(v)
            missing[n] += box is None
            x, y, s = brick_window((v.shape[1], v.shape[0]), box, args.side or None, args.margin)
            out = VCROPS / r["part"] / f"{Path(r['file']).stem}__{n}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), v[y:y + s, x:x + s], [cv2.IMWRITE_JPEG_QUALITY, 95])
            one = Path(f"{VCROPS}_{n}") / r["part"] / r["file"]  # this view only
            one.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(one), v[y:y + s, x:x + s], [cv2.IMWRITE_JPEG_QUALITY, 95])
    cfg = {"mode": "views", "side": args.side or None, "margin": args.margin, "frame": views["frame"],
           "views": views["views"], "empty_refs": {n: f"{n}_empty.png" for n in views["views"]}}
    (VCROPS / "crop.json").write_text(json.dumps(cfg))
    for n in views["views"]:
        (Path(f"{VCROPS}_{n}") / "crop.json").write_text(json.dumps({**cfg, "mode": "view", "view": n}))
    SPLIT_VIEWS.write_text(json.dumps(split, indent=1))
    counts = {}
    for r in split:
        counts[r["part"]] = counts.get(r["part"], 0) + 1
    print("photos:", counts, "| bricks:", {k: len(v) for k, v in by_brick.items()},
          "| defect types:", {t: sum(r.get("type") == t for r in split) for t in sorted({r.get('type') for r in split} - {None})})
    print(f"crops (3 per photo) -> {VCROPS}; no brick found (whole view used): {missing}")


def prepare_views_tagged(args) -> None:
    if args.out in ("", *load_views(VIEWS_PATH)["views"]):
        sys.exit(f"--out {args.out!r} would clash with the baseline / per-view crop folders")
    split = json.loads(SPLIT_VIEWS.read_text())
    views = load_views(VIEWS_PATH)
    refs = {n: EmptyCheck.load(VIEW_REFS / f"{n}_empty.png") for n in views["views"]}
    out_dir = Path(f"{VCROPS}_{args.out}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    missing, sides = {n: 0 for n in views["views"]}, []
    for r in split:
        for n, v in view_crops(cv2.imread(str(DATA / r["class"] / r["file"])), views).items():
            box = refs[n].brick_box(v)
            missing[n] += box is None
            x, y, s = brick_window((v.shape[1], v.shape[0]), box, args.side or None, args.margin)
            sides.append(s)
            out = out_dir / r["part"] / f"{Path(r['file']).stem}__{n}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), v[y:y + s, x:x + s], [cv2.IMWRITE_JPEG_QUALITY, 95])
    cfg = {"mode": "views", "side": args.side or None, "margin": args.margin, "frame": views["frame"],
           "views": views["views"], "empty_refs": {n: f"{n}_empty.png" for n in views["views"]}}
    (out_dir / "crop.json").write_text(json.dumps(cfg))
    counts = {}
    for r in split:
        counts[r["part"]] = counts.get(r["part"], 0) + 1
    print(f"photos (split_views.json): {counts}; crops (3 per photo, margin {args.margin}) -> {out_dir}; "
          f"square side median {int(np.median(sides))} px; no brick found (whole view used): {missing}")


def cmd_eval_views(args) -> None:
    """Score = max over the three views; threshold from the calibration goods (as eval); per-view AUROC too.

    --mask: each view's score is the max (--pct < 100: that percentile) of the anomaly map inside the brick outline
    grown by --dilate view px (MultiViewScorer mask), not the model's image score. Written to --out (default
    <name>_mask) with "base_model" and "mask", so the cell program loads it as its own --anomaly-model name.
    """
    mdir = next((MODELS / args.name).rglob("checkurai_meta.json")).parent
    meta = json.loads((mdir / "checkurai_meta.json").read_text())
    split = json.loads(SPLIT_VIEWS.read_text())
    mask = {"dilate": args.dilate, "pct": args.pct} if args.mask else None
    out_name = args.out or (f"{args.name}_mask" if mask else args.name)
    sc = MultiViewScorer(mdir, device=args.device, mask=mask)
    rows = []
    for r in split:
        if r["part"] == "train_good":
            continue
        img = cv2.imread(str(DATA / r["class"] / r["file"]))
        t = time.perf_counter()
        _, per, ms = sc(img)
        wall = (time.perf_counter() - t) * 1000
        rows.append({**r, "views": per, "ms": ms, "wall_ms": wall,
                     **({"model_views": {n: v["model_score"] for n, v in sc.last.items()}} if mask else {})})
    # each view on one scale: (s - median) / (max - median) of that view's calibration good scores
    cal_rows = [r for r in rows if r["part"] == "calib_good"]
    sc.norm = {n: [float(np.median([r["views"][n] for r in cal_rows])), float(max(r["views"][n] for r in cal_rows))]
               for n in sc.views["views"]}
    for r in rows:
        r["score"] = sc.combine(r["views"])
    get = lambda part, t=None: np.array([r["score"] for r in rows if r["part"] == part and (t is None or r.get("type") == t)])
    calib, test, defect = get("calib_good"), get("test_good"), get("defect")
    types = sorted({r["type"] for r in rows if r["part"] == "defect"})
    low = float(calib.max())
    high = float(low + (low - np.median(calib)))
    low2 = float(test.max())
    heldout = np.concatenate([calib, test])
    view_scores = lambda part, n: np.array([r["views"][n] for r in rows if r["part"] in part])
    res = {
        "name": out_name, **{k: meta[k] for k in ("model", "image_size", "normal_dirs", "train_seconds")},
        **({"base_model": args.name, "mask": sc.mask} if mask else {}),
        "counts": {"calib_good": len(calib), "test_good": len(test), "defect": len(defect)},
        "auroc_heldout_good_vs_defect": round(auroc(heldout, defect), 4),
        "auroc_test_good_vs_defect": round(auroc(test, defect), 4),
        "auroc_per_view": {n: round(auroc(view_scores(("calib_good", "test_good"), n), view_scores(("defect",), n)), 4)
                           for n in sc.views["views"]},
        "threshold": {"low": low, "high": high,
                      "rule": "score (normalised max over views, median over frames) < low: good; >= low: defect (low..high is flagged uncertain)",
                      "low_from": "max score of the calibration good photos", "high_from": "low + (low - median calibration good score)"},
        "confusion_at_low": {"defect_as_defect": int((defect >= low).sum()), "defect_as_good": int((defect < low).sum()),
                             "good_as_defect": int((test >= low).sum()), "good_as_good": int((test < low).sum())},
        "recall_per_type": {t: {"n": int(len(get("defect", t))), "recall": round(float((get("defect", t) >= low).mean()), 3),
                                "auroc": round(auroc(heldout, get("defect", t)), 4)} for t in types},
        "policy_uncertain_is_defect": {
            "false_alarm_rate_test": f"{int((test >= low).sum())}/{len(test)}",
            "false_alarm_rate_2fold": f"{int((test >= low).sum()) + int((calib >= low2).sum())}/{len(test) + len(calib)}",
            "recall_2fold": {t: round(float(((get('defect', t) >= low).mean() + (get('defect', t) >= low2).mean()) / 2), 3) for t in types},
            "recall_all_2fold": round(float(((defect >= low).mean() + (defect >= low2).mean()) / 2), 3)},
        "recall_at_high": {"all": f"{int((defect >= high).sum())}/{len(defect)}",
                           **{t: round(float((get("defect", t) >= high).mean()), 3) for t in types},
                           "test_good_at_high": int((test >= high).sum())},
        "latency_ms_3_views": {"device": sc.device, "median": round(float(np.median([r["ms"] for r in rows])), 1),
                               "wall_median": round(float(np.median([r["wall_ms"] for r in rows])), 1)},
        "score_rule": ("max over views of (raw - median) / (max - median) of that view's calibration good scores"
                       + ("; raw = anomaly map inside the brick outline" if mask else "")),
        "crop": meta.get("crop", {}).get("mode"),
        "view_norm": sc.norm,
        "auroc_max_raw": round(auroc(np.array([max(r["views"].values()) for r in rows if r["part"] in ("calib_good", "test_good")]),
                                     np.array([max(r["views"].values()) for r in rows if r["part"] == "defect"])), 4),
        "scores": {"calib_good": calib.round(4).tolist(), "test_good": test.round(4).tolist(),
                   "defect": {t: get("defect", t).round(4).tolist() for t in types}},
        "rows": [{k: r[k] for k in ("file", "part", "score", "views", "model_views") if k in r} | ({"type": r["type"]} if "type" in r else {})
                 for r in rows],
    }
    (RESULTS / f"{out_name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("scores", "rows")}, indent=1))


def cmd_viewset(args) -> None:
    """One model per view (train --crops views_<view>) -> a set the cell program loads as one model."""
    out = MODELS / f"patchcore_viewset_{args.size}" / "weights" / "openvino"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    views = load_views(VIEWS_PATH)
    models = {}
    for n in views["views"]:
        xml = next((MODELS / f"patchcore_views_{n}_{args.size}").rglob("*.xml"))
        models[n] = str(Path("..", "..", "..", xml.relative_to(MODELS)))
    for ref in VIEW_REFS.glob("*_empty.*"):
        shutil.copy(ref, out / ref.name)
    cfg = {"mode": "viewset", "frame": views["frame"], "views": views["views"], "models": models,
           "empty_refs": {n: f"{n}_empty.png" for n in views["views"]}, "margin": args.margin, "side": None}
    meta = {"name": out.parents[1].name, "model": "patchcore", "image_size": args.size, "normal_dirs": ["train_good"],
            "train_seconds": None, "crop": cfg}
    (out / "checkurai_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"viewset -> {out}: {models}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="", help="mirror-booth photo set tag (before the command), e.g. --set b2 views")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("roi")
    p.add_argument("--margin", type=int, default=24)
    sub.add_parser("prepare")
    p = sub.add_parser("prepare-brick")
    p.add_argument("--side", type=int, default=-1, help="crop side in ROI px; -1 auto (fits every brick), 0 tight box")
    p.add_argument("--margin", type=int, default=24)
    p = sub.add_parser("train")
    p.add_argument("--crops", default="", help="brick crop set from prepare-brick, e.g. brick330 (default: whole ROI)")
    p.add_argument("--model", choices=["patchcore", "efficientad"], required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--all-faces", action="store_true", help="also treat side / bottom faces as good")
    p.add_argument("--steps", type=int, default=2000, help="EfficientAd training steps")
    p.add_argument("--coreset", type=float, default=0.1)
    p.add_argument("--device", choices=["xpu", "cpu"], default="xpu")
    p.add_argument("--name", default="", help="model name (default: <model>_[<crops>_]<size>); a variant must not reuse an existing name")
    p = sub.add_parser("eval")
    p.add_argument("--name", required=True)
    p.add_argument("--no-latency", action="store_true")
    p = sub.add_parser("empty")
    p.add_argument("--pixel-thr", type=float, default=35.0)
    p.add_argument("--frac-thr", type=float, default=0.02)
    p = sub.add_parser("report")
    p.add_argument("--plot", nargs="*", default=[])
    sub.add_parser("sessions")
    p = sub.add_parser("screen")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("views")
    p.add_argument("--margin", type=int, default=40)
    p.add_argument("--pixel-thr", type=float, default=35.0)
    p = sub.add_parser("prepare-views")
    p.add_argument("--side", type=int, default=0, help="crop side in view px; 0 tight box around the brick")
    p.add_argument("--margin", type=int, default=40)
    p.add_argument("--out", default="", help="crop set tag -> data/inspection_crops_views_TAG (keeps split_views.json)")
    p.add_argument("--held-out", default="2,5,8", help="held-out photo indices (mod 10) per brick; the rest is training")
    p = sub.add_parser("viewset")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--margin", type=int, default=40)
    p = sub.add_parser("eval-views")
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="GPU")
    p.add_argument("--mask", action="store_true", help="score the anomaly map inside the brick outline only")
    p.add_argument("--dilate", type=int, default=MultiViewScorer.MASK_DEFAULTS["dilate"], help="grow the outline (view px)")
    p.add_argument("--pct", type=float, default=MultiViewScorer.MASK_DEFAULTS["pct"], help="percentile of the map; 100 = max")
    p.add_argument("--out", default="", help="results name (default: --name, or <name>_mask with --mask)")
    args = ap.parse_args()
    if args.set:
        use_set(args.set)
    {"roi": cmd_roi, "prepare": cmd_prepare, "prepare-brick": cmd_prepare_brick, "train": cmd_train, "eval": cmd_eval, "empty": cmd_empty,
     "report": cmd_report, "sessions": cmd_sessions, "screen": cmd_screen, "views": cmd_views,
     "prepare-views": cmd_prepare_views, "eval-views": cmd_eval_views, "viewset": cmd_viewset}[args.cmd](args)


if __name__ == "__main__":
    main()
