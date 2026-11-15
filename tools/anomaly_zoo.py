#!/usr/bin/env python3
"""Anomaly model zoo: other model families / backbones / sizes on the mirror-booth view crops, same protocol as the baseline.

    python tools/anomaly_zoo.py prepare-old          # old-session goods (data/inspection_b2/_old_good_1933) -> view crops
    python tools/anomaly_zoo.py run --model patchcore --backbone resnet18 --size 384      # train + eval (+ --mask eval)
    python tools/anomaly_zoo.py run --model padim --size 256 --old                         # + old-session goods in training
    python tools/anomaly_zoo.py train ... / eval --name NAME [--mask]                      # the two halves of run
    python tools/anomaly_zoo.py table                # results/anomaly_zoo/table.md + table.json (baseline rows first)
    python tools/anomaly_zoo.py promote --name zoo_x --as NEW_NAME    # -> models/anomaly/NEW_NAME, results/anomaly/NEW_NAME[_mask].json

Training data: data/inspection_crops_views_b2/train_good (the baseline's 44 photos x 3 views, split_views_b2.json); with
--old also data/inspection_crops_views_b2_zoo/train_good_old (prepare-old: the same crop code as prepare-views, i.e. the b2
view rectangles, the b2 empty references, tight box + 40 px margin). Held-out goods and defects are never trained on.

Evaluation is anomaly_train.cmd_eval_views itself (imported, not copied): the full b2 photos through MultiViewScorer, each
view normalised by the calibration goods, max over views, threshold = max calibration good score; it is run with the
module's MODELS / RESULTS pointed at models/anomaly_zoo and results/anomaly_zoo, so the baseline folders are not touched.
`eval` adds "latency_gpu_per_view" (anomaly_train.latency: one view, static shape, 30 runs) to that results file.
Candidates: models/anomaly_zoo/<name>/weights/openvino/{model.xml,bin, checkurai_meta.json, <view>_empty.png/json}, the
layout of models/anomaly, so `promote` only copies the folder and the results file under a new name and
AnomalibInspector (tools/orchestrator.py) loads it unchanged.

Sharing the laptop with the live cell: every job first waits (polling every 30 s, no time limit) while cell_live.py /
orchestrator.py / go_home.py runs, runs at nice 19 and is pinned to --cpus cores (default 4: OMP / torch threads, the
data loader workers and OpenVINO CPU all stay on them; CPU jobs on cores 0.., GPU jobs on the next block). GPU jobs also
wait (--wait s) while anomaly_train.py runs (another trainer) or /tmp/zoo_gpu_hold exists; if it is still busy they
run on the CPU and say so.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")  # before numpy / torch load their thread pools

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anomaly_train as at  # noqa: E402
from inspection import EmptyCheck, brick_window, load_views, view_crops  # noqa: E402

ROOT = at.ROOT
ZOO_MODELS = ROOT / "models" / "anomaly_zoo"
ZOO_RESULTS = ROOT / "results" / "anomaly_zoo"
BASELINES = ("patchcore_views_b2g_384", "patchcore_views_b2g_384_mask")
GPU_USERS = ("cell_live.py", "orchestrator.py", "anomaly_train.py")
CELL = ("cell_live.py", "orchestrator.py", "go_home.py")  # a live cell run: start nothing while one is going
GPU_HOLD = Path("/tmp/zoo_gpu_hold")  # touch it to keep the zoo off the GPU (e.g. while the arm is tested)
DEFAULT_BACKBONE = {"patchcore": "wide_resnet50_2", "padim": "resnet18", "rd": "wide_resnet50_2", "fastflow": "resnet18",
                    "stfpm": "resnet18", "efficientad": None, "anomalydino": "vit_small_patch14_dinov2"}
DEFAULT_LAYERS = {"patchcore": "layer2,layer3", "padim": "layer1,layer2,layer3", "rd": "layer1,layer2,layer3",
                  "stfpm": "layer1,layer2,layer3"}
SHORT = {"wide_resnet50_2": "wrn50", "resnet18": "r18", "resnet34": "r34", "resnet50": "r50",
         "vit_small_patch14_dinov2": "dinov2s"}


def setup(tag: str) -> None:
    """The b2 booth paths (anomaly_train.use_set), then results / models of the zoo (split file stays the set's)."""
    at.use_set(tag)
    global ZOO_CROPS, OLD_GOOD
    ZOO_CROPS = Path(f"{at.VCROPS}_zoo")
    OLD_GOOD = at.DATA / "_old_good_1933"


# ---------------------------------------------------------------- GPU sharing
def procs(names: tuple[str, ...]) -> list[str]:
    """Running python processes (not this one) whose command line names one of these scripts."""
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if argv and b"python" in Path(argv[0].decode(errors="ignore")).name.encode():
            line = " ".join(a.decode(errors="ignore") for a in argv)
            if any(u in line for u in names):
                found.append(f"{pid}: {line[:120]}")
    return found


def gpu_users() -> list[str]:
    """Running processes that use the GPU for live cycles or training (not this one); GPU_HOLD = held by hand."""
    return ([f"hold file {GPU_HOLD}"] if GPU_HOLD.exists() else []) + procs(GPU_USERS)


def cell_quiet() -> None:
    """Wait (no limit, poll 30 s) until no live cell run is going: the cell must not compete for compute."""
    while busy := procs(CELL):
        print(f"[zoo] {time.strftime('%H:%M:%S')} cell run going, waiting: {busy[0]}", flush=True)
        time.sleep(30)


def limit_cpu(cpus: int, gpu_job: bool) -> None:
    """nice 19, pinned to `cpus` cores (CPU jobs 0..cpus-1, GPU jobs the next block), torch threads = cpus."""
    os.nice(19 - os.nice(0))
    n = os.cpu_count() or cpus
    first = cpus if gpu_job and 2 * cpus <= n else 0
    os.sched_setaffinity(0, set(range(first, first + cpus)))
    import torch

    torch.set_num_threads(cpus)


def gpu_free(wait_s: float) -> bool:
    cell_quiet()
    t0 = time.time()
    while True:
        busy = gpu_users()
        if not busy:
            return True
        if time.time() - t0 >= wait_s:
            print(f"[zoo] GPU still in use after {wait_s:.0f}s: {busy}", flush=True)
            return False
        print(f"[zoo] waiting for the GPU: {busy[0]}", flush=True)
        time.sleep(10)


# ---------------------------------------------------------------- old-session goods
def cmd_prepare_old(args) -> None:
    """View crops of the old-session good photos, made exactly like prepare-views (b2 views, b2 empty refs, crop.json)."""
    out = ZOO_CROPS / "train_good_old"
    if out.exists():
        sys.exit(f"{out} exists; remove it yourself to remake it")
    views = load_views(at.VIEWS_PATH)
    cfg = json.loads((at.VCROPS / "crop.json").read_text())
    refs = {n: EmptyCheck.load(at.VIEW_REFS / f"{n}_empty.png") for n in views["views"]}
    photos = sorted(OLD_GOOD.glob("*.jpg"))
    missing, spot, tiles, skipped = {n: 0 for n in views["views"]}, [], [], []
    out.mkdir(parents=True)
    for p in photos:
        img = cv2.imread(str(p))
        vc = view_crops(img, views)
        if not refs["direct"].brick_present(vc["direct"]):  # no brick on the spot (photo taken while swapping)
            skipped.append(p.name)
            continue
        row = []
        for n, v in vc.items():
            box = refs[n].brick_box(v)
            missing[n] += box is None
            x, y, s = brick_window((v.shape[1], v.shape[0]), box, cfg.get("side"), cfg.get("margin", 40))
            sq = v[y:y + s, x:x + s]
            cv2.imwrite(str(out / f"{p.stem}__{n}.jpg"), sq, [cv2.IMWRITE_JPEG_QUALITY, 95])
            row.append(cv2.resize(sq, (96, 96), interpolation=cv2.INTER_AREA))
            if n == "direct":
                spot.append(refs[n].changed_fraction(v))
        tiles.append(np.hstack(row))
    (ZOO_CROPS / "crop.json").write_text(json.dumps(cfg))
    ZOO_RESULTS.mkdir(parents=True, exist_ok=True)
    cols = 6
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(ZOO_RESULTS / "old_goods_montage.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    fr = np.array(spot)
    n = len(photos) - len(skipped)
    print(f"{n} old good photos -> {out} ({3 * n} crops); skipped (no brick on the spot): {skipped}; "
          f"no brick found in a view (whole view used): {missing}; "
          f"direct-view changed fraction min {fr.min():.3f} median {np.median(fr):.3f} (spot thr {refs['direct'].frac_thr}); "
          f"montage {ZOO_RESULTS / 'old_goods_montage.jpg'}")


# ---------------------------------------------------------------- training
def auto_name(args) -> str:
    bb = args.backbone or DEFAULT_BACKBONE[args.model]
    parts = ["zoo"] + ([args.set] if args.set != "b2" else []) + [args.model]
    if bb:
        parts.append(SHORT.get(bb, bb.replace("_", "")))
    if args.layers and args.layers != DEFAULT_LAYERS.get(args.model):
        parts.append("l" + "".join(x[-1] for x in args.layers.split(",")))
    if args.model == "patchcore" and args.coreset != 0.1:
        parts.append(f"c{args.coreset:g}".replace(".", ""))
    parts.append(str(args.size))
    if args.old:
        parts.append("old")
    if args.augment:
        parts.append("aug")
    return "_".join(parts)


def light_augment():
    """Training-only photometric jitter (global light / white-balance / contrast shifts) + horizontal flip, so a trained
    family (STFPM, RD, FastFlow) learns that these are normal; PatchCore / PaDiM ignore it (one pass, no epochs)."""
    from torchvision.transforms import v2

    return v2.Compose([v2.RandomHorizontalFlip(0.5),
                       v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)])


def build_model(args):
    from anomalib import models as M

    size = (args.size, args.size)
    bb = args.backbone or DEFAULT_BACKBONE[args.model]
    layers = (args.layers or DEFAULT_LAYERS.get(args.model, "")).split(",")
    common = {"post_processor": False, "evaluator": False, "visualizer": False}
    if args.model == "patchcore":
        return M.Patchcore(backbone=bb, layers=layers, coreset_sampling_ratio=args.coreset, num_neighbors=args.neighbors,
                           pre_processor=M.Patchcore.configure_pre_processor(image_size=size), **common)
    if args.model == "padim":
        return M.Padim(backbone=bb, layers=layers, n_features=args.n_features or None,
                       pre_processor=M.Padim.configure_pre_processor(image_size=size), **common)
    if args.model == "rd":
        return M.ReverseDistillation(backbone=bb, layers=layers,
                                     pre_processor=M.ReverseDistillation.configure_pre_processor(image_size=size), **common)
    if args.model == "fastflow":
        return M.Fastflow(backbone=bb, flow_steps=8, pre_processor=M.Fastflow.configure_pre_processor(image_size=size), **common)
    if args.model == "stfpm":
        return M.Stfpm(backbone=bb, layers=layers, pre_processor=M.Stfpm.configure_pre_processor(image_size=size), **common)
    if args.model == "anomalydino":  # training-free DINOv2 patch kNN (size: a multiple of 14)
        return M.AnomalyDINO(num_neighbours=1, encoder_name=bb, masking=False, coreset_subsampling=args.coreset < 1,
                             sampling_ratio=args.coreset,
                             pre_processor=M.AnomalyDINO.configure_pre_processor(image_size=size), **common)
    if args.model == "efficientad":
        return M.EfficientAd(imagenet_dir=at.IMAGENETTE, model_size=args.ead_size,
                             pre_processor=M.EfficientAd.configure_pre_processor(image_size=size), **common)
    raise SystemExit(f"unknown model {args.model}")


def cmd_train(args) -> str:
    from anomalib.data import Folder
    from anomalib.deploy import ExportType

    name = args.name or auto_name(args)
    out = ZOO_MODELS / name
    if out.exists():
        if not args.retrain:
            print(f"[zoo] {name} exists, not retrained (--retrain to redo it)")
            return name
        shutil.rmtree(out)
    device = args.device
    if device == "xpu" and not gpu_free(args.wait):
        device = "cpu"
        print("[zoo] training on the CPU instead", flush=True)
    root = at.VCROPS
    crop_cfg = json.loads((root / "crop.json").read_text())
    normal = ["train_good"] + ([str(ZOO_CROPS / "train_good_old")] if args.old else [])
    one_pass = args.model in ("patchcore", "padim", "anomalydino")
    ead = args.model == "efficientad"
    dm = Folder(name=name, root=root, normal_dir=normal, abnormal_dir="defect", normal_test_dir="test_good",
                train_batch_size=1 if ead else args.batch, eval_batch_size=8, num_workers=4,
                val_split_mode="from_train" if ead else "none", val_split_ratio=0.15, seed=0,
                train_augmentations=light_augment() if args.augment else None)
    model = build_model(args)
    kw = {"default_root_dir": str(out / "work")}
    if one_pass:
        kw.update(max_epochs=1, limit_val_batches=0)
    elif ead:
        kw.update(max_steps=args.steps, check_val_every_n_epoch=1000)
    else:
        kw.update(max_epochs=args.epochs, limit_val_batches=0, num_sanity_val_steps=0)
    engine = at.engine_for(device, **kw)
    t0 = time.time()
    engine.fit(model=model, datamodule=dm)
    if ead:
        engine.validate(model=model, datamodule=dm)  # map-normalisation quantiles from the final weights
    train_s = time.time() - t0
    size = (args.size, args.size)
    engine.export(model=model, export_type=ExportType.OPENVINO, export_root=out, input_size=size)
    xml = next(out.rglob("*.xml"))
    meta = {"name": name, "model": args.model, "image_size": args.size,
            "normal_dirs": ["train_good"] + (["train_good_old (data/inspection_crops_views_b2_zoo)"] if args.old else []),
            "train_seconds": round(train_s, 1), "device": device, "xml": str(xml.relative_to(out)),
            "backbone": args.backbone or DEFAULT_BACKBONE[args.model],
            "layers": (args.layers or DEFAULT_LAYERS.get(args.model, "")).split(",") if args.model != "efficientad" else None,
            "train_images": len(dm.train_data), "photo_set": args.set, "augment": bool(args.augment),
            "made_by": "tools/anomaly_zoo.py"}
    if args.model == "patchcore":
        meta.update(coreset=args.coreset, neighbors=args.neighbors)
    elif args.model == "padim":
        meta["n_features"] = args.n_features or None
    elif ead:
        meta.update(steps=args.steps, ead_size=args.ead_size)
    else:
        meta["epochs"] = args.epochs
    meta["crop"] = crop_cfg  # the scorer finds the brick in each view with these empty references
    for ref in crop_cfg["empty_refs"].values():
        for ext in (".png", ".json"):
            src = (at.VIEW_REFS / ref).with_suffix(ext)
            shutil.copy(src, xml.parent / src.name)
    (xml.parent / "checkurai_meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(out / "work", ignore_errors=True)
    print(f"[zoo] trained {name} in {train_s:.0f}s on {device} ({meta['train_images']} train crops) -> {xml}", flush=True)
    return name


# ---------------------------------------------------------------- evaluation
def cmd_eval(args) -> None:
    device = args.eval_device
    if device == "GPU" and not gpu_free(args.wait):
        device = "CPU"
        print("[zoo] evaluating on the CPU instead (latency numbers are CPU)", flush=True)
    # --ref: a model of models/anomaly (the baseline) re-evaluated the same way, written as zoo_ref_<name>[_mask]
    src = ROOT / "models" / "anomaly" if args.ref else ZOO_MODELS
    at.MODELS, at.RESULTS = src, ZOO_RESULTS  # split file (SPLIT_VIEWS) was already set by use_set
    ZOO_RESULTS.mkdir(parents=True, exist_ok=True)
    prefix = f"zoo_ref_{args.name}" if args.ref else args.name
    outs = []
    for mask in ([False, True] if args.mask == "both" else [args.mask == "yes"]):
        outs.append(prefix + ("_mask" if mask else ""))
        ns = argparse.Namespace(name=args.name, device=device, mask=mask, exposure=False, out=outs[-1],
                                dilate=at.MultiViewScorer.MASK_DEFAULTS["dilate"], pct=at.MultiViewScorer.MASK_DEFAULTS["pct"],
                                tight=False)
        at.cmd_eval_views(ns)
    xml = next((src / args.name).rglob("*.xml"))
    size = json.loads((xml.parent / "checkurai_meta.json").read_text())["image_size"]
    lat = None
    if args.latency and (device == "GPU" and gpu_free(args.wait)):
        lat = {**at.latency(xml, size, "GPU"), "contended_by": gpu_users() or None}
    for out in outs:
        p = ZOO_RESULTS / f"{out}.json"
        r = json.loads(p.read_text())
        if lat:
            r["latency_gpu_per_view"] = lat
        r["eval_device"] = device
        p.write_text(json.dumps(r, indent=1))
    print(f"[zoo] evaluated {args.name} on {device}; per-view GPU latency {lat}", flush=True)


def cmd_run(args) -> None:
    args.name = cmd_train(args)
    cmd_eval(args)
    cmd_table(args)


# ---------------------------------------------------------------- table
def summarise(r: dict) -> dict:
    pol = r["policy_uncertain_is_defect"]
    c = r["confusion_at_low"]
    lat = r.get("latency_gpu_per_view") or {}
    lat3 = r.get("latency_ms_3_views") or {}
    return {"name": r["name"], "model": r["model"], "size": r["image_size"], "train_s": r.get("train_seconds"),
            "auroc": r["auroc_heldout_good_vs_defect"], "auroc_test": r["auroc_test_good_vs_defect"],
            "auroc_per_view": r.get("auroc_per_view"),
            "recall": f"{c['defect_as_defect']}/{c['defect_as_defect'] + c['defect_as_good']}",
            "false_alarms": pol["false_alarm_rate_test"], "false_alarms_2fold": pol["false_alarm_rate_2fold"],
            "recall_2fold": pol["recall_all_2fold"], "recall_at_high": r.get("recall_at_high", {}).get("all"),
            "ms_view_gpu": lat.get("median_ms"),
            "ms_3views_eval": lat3.get("median"), "eval_device": lat3.get("device")}


def auroc_diff_ci(cand: list[dict], base: list[dict], n: int = 2000) -> list[float] | None:
    """90% paired bootstrap interval of AUROC(candidate) - AUROC(baseline) over the same held-out photos (goods and
    defects resampled separately); None when the photo lists differ."""
    sc = lambda rows: {r["file"]: (r["score"], r["part"] == "defect") for r in rows if r["part"] != "train_good"}
    a, b = sc(cand), sc(base)
    if set(a) != set(b):
        return None
    files = sorted(a)
    good = np.array([f for f in files if not a[f][1]])
    bad = np.array([f for f in files if a[f][1]])
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n):
        g, d = rng.choice(good, len(good)), rng.choice(bad, len(bad))
        diffs.append(at.auroc(np.array([a[f][0] for f in g]), np.array([a[f][0] for f in d]))
                     - at.auroc(np.array([b[f][0] for f in g]), np.array([b[f][0] for f in d])))
    return [round(float(np.percentile(diffs, 5)), 3), round(float(np.percentile(diffs, 95)), 3)]


def cmd_table(args) -> None:
    lat = json.loads((ZOO_RESULTS / "latency.json").read_text()) if (ZOO_RESULTS / "latency.json").exists() else {}
    rows, full = [], {}
    res_dir = at.ROOT / "results" / "anomaly"
    for p in [res_dir / f"{b}.json" for b in BASELINES] + sorted(res_dir.glob("patchcore_views_b2all*.json")):
        if p.exists():
            full[p.stem] = json.loads(p.read_text())
            rows.append({**summarise(full[p.stem]), "baseline": True})
    for p in sorted(ZOO_RESULTS.glob("zoo_*.json")):
        full[p.stem] = json.loads(p.read_text())
        rows.append(summarise(full[p.stem]))
    for r in rows:
        r["set"] = "b2all" if "b2all" in r["name"] else "b2"
        base = r["name"].removeprefix("zoo_ref_").removesuffix("_mask")
        if r["ms_view_gpu"] is None and base in lat and lat[base].get("ok"):
            r["ms_view_gpu"] = lat[base]["median_ms"]
        # paired bootstrap against the set's baseline with the same scoring (plain vs plain, mask vs mask)
        ref = BASELINES[1] if r["name"].endswith("_mask") else BASELINES[0]
        if r["set"] == "b2all":
            ref = getattr(args, "b2all_ref", "patchcore_views_b2all_fx_384") + ("_mask" if r["name"].endswith("_mask") else "")
        r["vs"] = ref
        r["auroc_diff_ci90"] = (auroc_diff_ci(full[r["name"]]["rows"], full[ref]["rows"])
                                if ref in full and r["name"] != ref and "rows" in full[r["name"]] else None)
    rows.sort(key=lambda r: (r["set"], not r.get("baseline"), r["name"]))
    ZOO_RESULTS.mkdir(parents=True, exist_ok=True)
    (ZOO_RESULTS / "table.json").write_text(json.dumps(rows, indent=1))
    head = ("| set | model | size | AUROC held-out | AUROC diff vs baseline, 90% CI | recall @thr | false alarms test | FA 2-fold | "
            "recall 2-fold | ms/view GPU | ms 3 views (eval dev) | train s |")
    lines = [head, "|" + "---|" * 12]
    for r in rows:
        ci = r["auroc_diff_ci90"]
        lines.append(f"| {r['set']} | {r['name']}{' (baseline)' if r.get('baseline') else ''} | {r['size']} | {r['auroc']:.3f} | "
                     f"{f'{ci[0]:+.3f} .. {ci[1]:+.3f}' if ci else '-'} | "
                     f"{r['recall']} | {r['false_alarms']} | {r['false_alarms_2fold']} | {r['recall_2fold']:.3f} | "
                     f"{r['ms_view_gpu'] or '-'} | {r['ms_3views_eval']} ({r['eval_device']}) | {r['train_s']} |")
    (ZOO_RESULTS / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_latency(args) -> None:
    """Per-view GPU latency (anomaly_train.latency, one view, 30 runs) of zoo models / models/anomaly models, one at a
    time on a quiet GPU -> results/anomaly_zoo/latency.json (the table uses it where an eval had no GPU number)."""
    p = ZOO_RESULTS / "latency.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    for name in args.names:
        if not gpu_free(args.wait):
            print(f"[zoo] GPU busy, latency of {name} not measured")
            continue
        mdir = ZOO_MODELS / name if (ZOO_MODELS / name).exists() else ROOT / "models" / "anomaly" / name
        xml = next(mdir.rglob("*.xml"))
        size = json.loads((xml.parent / "checkurai_meta.json").read_text())["image_size"]
        out[name] = {**at.latency(xml, size, args.device), "device": args.device, "at": time.strftime("%H:%M:%S")}
        print(f"[zoo] {name}: {out[name]}", flush=True)
        p.write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------- promote
def cmd_promote(args) -> None:
    """Copy a zoo model + its results under a new name into models/anomaly and results/anomaly (never overwrites)."""
    src, dst = ZOO_MODELS / args.name, ROOT / "models" / "anomaly" / args.to
    res_dir = ROOT / "results" / "anomaly"
    targets = [res_dir / f"{args.to}.json", res_dir / f"{args.to}_mask.json"]
    if dst.exists() or any(t.exists() for t in targets):
        sys.exit(f"{dst} or its results exist already; pick another name")
    shutil.copytree(src, dst)
    meta_p = next(dst.rglob("checkurai_meta.json"))
    meta = json.loads(meta_p.read_text())
    meta.update(name=args.to, zoo_name=args.name)
    meta_p.write_text(json.dumps(meta, indent=2))
    for suffix, t in zip(("", "_mask"), targets, strict=True):
        p = ZOO_RESULTS / f"{args.name}{suffix}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        r.update(name=f"{args.to}{suffix}", zoo_name=f"{args.name}{suffix}")
        if "base_model" in r:
            r["base_model"] = args.to
        t.write_text(json.dumps(r, indent=1))
        print(f"results -> {t}")
    print(f"model -> {dst} (cell: --anomaly-model {args.to} or {args.to}_mask)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="b2", help="mirror-booth photo set tag (anomaly_train --set)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare-old")
    for cmd in ("train", "run", "eval"):
        p = sub.add_parser(cmd)
        p.add_argument("--wait", type=float, default=600, help="s to wait for a busy GPU before falling back to the CPU")
        if cmd in ("train", "run"):
            p.add_argument("--model", choices=list(DEFAULT_BACKBONE), required=True)
            p.add_argument("--backbone", default="", help="timm backbone (default per model); must be cached, no downloads")
            p.add_argument("--layers", default="", help="comma list, e.g. layer1,layer2,layer3")
            p.add_argument("--size", type=int, default=256)
            p.add_argument("--old", action="store_true", help="also train on the old-session goods (prepare-old)")
            p.add_argument("--augment", action="store_true", help="photometric jitter + flip in training (light_augment)")
            p.add_argument("--coreset", type=float, default=0.1)
            p.add_argument("--neighbors", type=int, default=9)
            p.add_argument("--n-features", type=int, default=0, help="PaDiM random feature subset (0: anomalib default)")
            p.add_argument("--epochs", type=int, default=50, help="RD / FastFlow / STFPM epochs")
            p.add_argument("--steps", type=int, default=2000, help="EfficientAd steps")
            p.add_argument("--ead-size", default="small", choices=["small", "medium"])
            p.add_argument("--batch", type=int, default=8)
            p.add_argument("--device", choices=["xpu", "cpu"], default="xpu")
            p.add_argument("--retrain", action="store_true")
            p.add_argument("--name", default="")
        else:
            p.add_argument("--name", required=True)
        if cmd in ("eval", "run"):
            p.add_argument("--eval-device", default="GPU")
            p.add_argument("--mask", choices=["no", "yes", "both"], default="both")
            p.add_argument("--no-latency", dest="latency", action="store_false")
            p.add_argument("--ref", action="store_true", help="eval: a model of models/anomaly -> results zoo_ref_<name>")
    p = sub.add_parser("table")
    p.add_argument("--b2all-ref", default="patchcore_views_b2all_fx_384", help="results/anomaly baseline of the b2all rows")
    p = sub.add_parser("latency")
    p.add_argument("names", nargs="+")
    p.add_argument("--device", default="GPU")
    p.add_argument("--wait", type=float, default=600)
    for p in [sub.choices[c] for c in ("train", "run", "eval", "latency")]:
        p.add_argument("--cpus", type=int, default=4, help="cores (and threads) this job may use, at nice 19")
    p = sub.add_parser("promote")
    p.add_argument("--name", required=True)
    p.add_argument("--as", dest="to", required=True)
    args = ap.parse_args()
    setup(args.set)
    if args.cmd in ("train", "run", "eval", "latency"):
        cell_quiet()
        gpu_job = (getattr(args, "device", "") == "xpu" or (args.cmd == "eval" and args.eval_device == "GPU")
                   or args.cmd == "latency")
        limit_cpu(args.cpus, gpu_job)
    {"prepare-old": cmd_prepare_old, "train": cmd_train, "eval": cmd_eval, "run": cmd_run, "table": cmd_table,
     "latency": cmd_latency, "promote": cmd_promote}[args.cmd](args)


if __name__ == "__main__":
    main()
