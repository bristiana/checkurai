#!/usr/bin/env python3
"""Supervised good / defect brick classifier: frozen foundation-model features (DINOv2 ViT-S/14) + a linear head.

    python tools/brick_classifier_fm.py train --name fm_dinov2s_224          # features, synthetic defects, head
    python tools/brick_classifier_fm.py export --name fm_dinov2s_224         # backbone -> OpenVINO IR + GPU/NPU/CPU ms
    python tools/brick_classifier_fm.py eval --name fm_dinov2s_224           # held-out goods + real defect photos
    python tools/brick_classifier_fm.py live --name fm_dinov2s_224           # tonight's live cycles (data/cell_runs)

    from tools.brick_classifier_fm import BrickClassifierFM
    clf = BrickClassifierFM("fm_dinov2s_224", device="GPU"); r = clf.inspect(frames)   # r.verdict / r.score / r.uncertain

At run time (BrickClassifierFM, eval, live) the backbone is the OpenVINO IR backbone.xml on --device (GPU default,
NPU or CPU); PyTorch is only used for training and as the export reference. The head is a numpy dot product.

Crops are the b2all crop format (data/inspection_crops_views_b2all/PREPROC.md), made at run time exactly like
inspection.MultiViewScorer does for a fill model: each view rectangle is brought to its empty reference's exposure,
the brick is the tight colour-hull outline (+6 px), a square around it (+40 px) is cut and everything outside the
outline is grey 128. The empty-spot check is MultiViewScorer.spot for a fill model (colour-hull fraction of the
direct view), with the direct empty reference's frac_thr.

Features per view crop (224 px): DINOv2 ViT-S/14 CLS token and the mean of the patch tokens that lie on the brick
(grey-fill patches left out), each L2-normalised, concatenated (768). Head: standardise + logistic regression.
Training: the b2all train goods (3 sessions) with photometric augmentation (brightness +-40 %, white balance, specular
highlight blobs; all labelled good) and synthetic defects (1-6 dark marker strokes / dots / scribbles inside the brick
outline, width 2-8 px, alpha 0.6-1, slight blur). The real defect photos are only used for validation (unless
--real-defects half). A photo's score = max over its three views of P(defect); one fixed threshold.
Model files: models/classifier/<name>/{head.npz, config.json, backbone.xml/bin (OpenVINO), backbone.pth (torch reference),
<view>_empty.png/json, synth_examples.jpg}; numbers in results/classifier/<name>.json and <name>_live.json.
Heavy jobs share the laptop with the live cell: run them through /tmp/cellsafe.sh (freezes them during a cell run);
the script also waits for a quiet cell at start, runs at nice 19 and on --cpus cores.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspection import EmptyCheck, prep_view, view_crops  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "data" / "inspection_crops_views_b2all"
PHOTOS = ROOT / "data" / "inspection_b2all"
SPLIT = ROOT / "results" / "anomaly" / "split_views_b2all.json"
REFS = ROOT / "models" / "anomaly" / "views_b2all"
OUT = ROOT / "models" / "classifier"
RESULTS = ROOT / "results" / "classifier"
ARCH = "vit_small_patch14_dinov2.lvd142m"
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
GREY = 128
# tonight's live cycles: the two blue 2x2 marker-scribbled cycles are defects, the rest clean (as /tmp/live_eval.py)
LIVE_DEFECT = {("20261114-202926", 0), ("20261114-203554", 0)}


def cell_quiet() -> None:
    """Wait (poll 30 s) while a live cell run (cell_live.py / orchestrator.py / go_home.py) is going."""
    names = (b"cell_live.py", b"orchestrator.py", b"go_home.py")
    while True:
        busy = []
        for pid in os.listdir("/proc"):
            if pid.isdigit() and int(pid) != os.getpid():
                try:
                    argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                except OSError:
                    continue
                if argv and b"python" in Path(argv[0].decode(errors="ignore")).name.encode() and any(
                        n in a for a in argv for n in names):
                    busy.append(pid)
        if not busy:
            return
        print(f"[fm] {time.strftime('%H:%M:%S')} cell run going (pid {busy[0]}), waiting", flush=True)
        time.sleep(30)


def limit_cpu(cpus: int) -> None:
    """nice 19 and `cpus` cores / threads: the live cell must not compete for compute."""
    os.nice(19 - os.nice(0))
    os.sched_setaffinity(0, set(range(cpus)))
    import torch

    torch.set_num_threads(cpus)


# ---------------------------------------------------------------- features
class Features:
    """Frozen DINOv2 ViT-S/14: (CLS, mean of the brick patch tokens), each L2-normalised -> 768-d.

    device "torch": timm in PyTorch on the CPU (training only). Otherwise the OpenVINO IR backbone.xml (export) on that
    OpenVINO device (GPU / NPU / CPU), batch 1, static shape: what the cell runs."""

    def __init__(self, size: int = 224, weights: Path | None = None, device: str = "torch", xml: Path | None = None):
        self.size, self.device, self.ms = size, device, []
        self.prefix = 1  # vit_small_patch14_dinov2: one CLS token, no registers
        if device != "torch":
            import openvino as ov

            core = ov.Core()
            self.compiled = core.compile_model(core.read_model(str(xml)), device)
            self.req = self.compiled.create_infer_request()
            return
        import timm
        import torch

        self.torch = torch
        if weights and Path(weights).exists():
            self.net = timm.create_model(ARCH, pretrained=False, img_size=size, num_classes=0)
            self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        else:  # public weights from the HF hub (no token)
            self.net = timm.create_model(ARCH, pretrained=True, img_size=size, num_classes=0)
        self.net.eval()
        self.prefix = getattr(self.net, "num_prefix_tokens", 1)

    def tokens(self, x: np.ndarray) -> np.ndarray:
        if self.device == "torch":
            with self.torch.inference_mode():
                return self.net.forward_features(self.torch.from_numpy(x)).numpy()
        out = []
        for i in range(len(x)):  # static batch 1 (NPU)
            t = time.perf_counter()
            out.append(np.asarray(self.req.infer({0: x[i:i + 1]})[self.compiled.outputs[0]]))
            self.ms.append((time.perf_counter() - t) * 1000)
        return np.concatenate(out)

    def __call__(self, crops: list[np.ndarray], batch: int = 16) -> np.ndarray:
        s, out = self.size, []
        g = s // 14
        for i in range(0, len(crops), batch):
            ims, masks = [], []
            for c in crops[i:i + batch]:
                im = cv2.resize(c, (s, s), interpolation=cv2.INTER_AREA)
                masks.append(cv2.resize((np.abs(im.astype(np.int16) - GREY).max(axis=2) > 12).astype(np.float32), (g, g),
                                        interpolation=cv2.INTER_AREA) > 0.2)
                ims.append(((cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))
            tok = self.tokens(np.stack(ims).astype(np.float32))
            cls, pat = tok[:, 0], tok[:, self.prefix:]
            m = np.stack(masks).reshape(len(masks), -1, 1).astype(np.float32)
            m[m.sum(axis=1)[:, 0] == 0] = 1.0  # no brick patch found: all patches
            mean = (pat * m).sum(axis=1) / m.sum(axis=1)
            nz = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-6)
            out.append(np.concatenate([nz(cls), nz(mean)], axis=1))
        return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------- synthetic data
def brick_mask(crop: np.ndarray) -> np.ndarray:
    """The brick inside a b2all crop: everything that is not the grey fill."""
    m = (np.abs(crop.astype(np.int16) - GREY).max(axis=2) > 12).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m > 0


def photometric(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator, highlight_p: float = 0.5) -> np.ndarray:
    """Brightness +-40 %, white-balance gains, sometimes 1-3 specular highlight blobs; only on the brick (grey stays)."""
    f = img.astype(np.float32) * rng.uniform(0.6, 1.4) * rng.uniform(0.85, 1.15, 3)[None, None, :]
    if rng.random() < highlight_p and mask.any():
        ys, xs = np.nonzero(mask)
        h, w = mask.shape
        for _ in range(rng.integers(1, 4)):
            k = rng.integers(len(ys))
            blob = np.zeros((h, w), np.float32)
            ax = (int(rng.uniform(0.02, 0.12) * w), int(rng.uniform(0.01, 0.06) * h))
            cv2.ellipse(blob, (int(xs[k]), int(ys[k])), (max(ax[0], 2), max(ax[1], 1)), rng.uniform(0, 180), 0, 360, 1.0, -1)
            blob = cv2.GaussianBlur(blob, (0, 0), max(ax) * 0.4 + 1) * rng.uniform(0.5, 1.0) * mask
            f = f * (1 - blob[..., None]) + 255.0 * blob[..., None]
    out = np.clip(f, 0, 255).astype(np.uint8)
    out[~mask] = img[~mask]
    return out


def marker(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """1-6 dark marker strokes (lines / Bezier scribbles / dots) inside the brick outline."""
    h, w = mask.shape
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if inner.sum() < 50:
        inner = mask
    ys, xs = np.nonzero(inner)
    if len(ys) == 0:
        return img
    layer = np.zeros((h, w), np.float32)
    scale = max(h, w) / 300.0
    for _ in range(rng.integers(1, 7)):
        k = rng.integers(len(ys))
        p0 = np.array([xs[k], ys[k]], np.float32)
        width = max(1, int(round(rng.uniform(2, 8) * scale)))
        kind = rng.choice(["line", "scribble", "dot"], p=[0.4, 0.3, 0.3])
        if kind == "dot":
            cv2.circle(layer, (int(p0[0]), int(p0[1])), max(2, int(rng.uniform(2, 9) * scale)), 1.0, -1)
            continue
        n = 2 if kind == "line" else rng.integers(3, 6)
        pts = [p0]
        for _ in range(n):
            step = rng.uniform(0.05, 0.25) * max(h, w)
            ang = rng.uniform(0, 2 * np.pi)
            pts.append(pts[-1] + step * np.array([np.cos(ang), np.sin(ang)], np.float32))
        t = np.linspace(0, 1, 30)[:, None]
        curve = []
        for a, b, c in zip(pts[:-2] or [pts[0]], pts[1:-1] or [pts[0]], pts[2:] or [pts[-1]], strict=False):
            curve.append((1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c)  # quadratic Bezier through the walk
        curve = np.concatenate(curve) if kind == "scribble" else np.array(pts)
        cv2.polylines(layer, [curve.astype(np.int32).reshape(-1, 1, 2)], False, 1.0, width, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (3, 3), 0) * mask * rng.uniform(0.6, 1.0)
    ink = rng.uniform(5, 60) + rng.uniform(-8, 8, 3)
    out = img.astype(np.float32) * (1 - layer[..., None]) + ink[None, None, :] * layer[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- data
def split_rows() -> list[dict]:
    return json.loads(SPLIT.read_text())


def crop_path(r: dict, view: str) -> Path:
    return CROPS / r["part"] / f"{Path(r['file']).stem}__{view}.jpg"


def cmd_train(args) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    rows = split_rows()
    views = ("direct", "left", "right")
    goods = [cv2.imread(str(crop_path(r, v))) for r in rows if r["part"] == "train_good" for v in views]
    goods = [g for g in goods if g is not None]
    X_img, y, kind = [], [], []
    for g in goods:
        m = brick_mask(g)
        X_img.append(g)
        y.append(0)
        kind.append("good")
        for _ in range(args.aug):
            X_img.append(photometric(g, m, rng))
            y.append(0)
            kind.append("good_aug")
        for _ in range(args.synth):
            X_img.append(photometric(marker(g, m, rng), m, rng, highlight_p=0.3))
            y.append(1)
            kind.append("synth_defect")
    real_used = []
    if args.real_defects == "half":  # every other real defect photo (by time), all three views
        drows = sorted([r for r in rows if r["part"] == "defect"], key=lambda r: r["file"])
        for r in drows[0::2]:
            real_used.append(r["file"])
            for v in views:
                X_img.append(cv2.imread(str(crop_path(r, v))))
                y.append(1)
                kind.append("real_defect")
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "synth_examples.jpg"), np.vstack([
        np.hstack([cv2.resize(X_img[i], (160, 160)) for i in range(j, j + 1 + args.aug + args.synth)])
        for j in range(0, 8 * (1 + args.aug + args.synth), 1 + args.aug + args.synth)]))
    t0 = time.time()
    fx = Features(args.size)
    X = fx(X_img)
    feat_s = time.time() - t0
    y = np.array(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, class_weight="balanced", max_iter=5000))
    clf.fit(X, y)
    sc, lr = clf[0], clf[1]
    np.savez(out / "head.npz", mean=sc.mean_, scale=sc.scale_, coef=lr.coef_[0], intercept=lr.intercept_)
    import torch

    torch.save(fx.net.state_dict(), out / "backbone.pth")
    crop_cfg = json.loads((CROPS / "crop.json").read_text())
    for n in crop_cfg["empty_refs"].values():
        for ext in (".png", ".json"):
            shutil.copy((REFS / n).with_suffix(ext), out / Path(n).with_suffix(ext).name)
    cfg = {"name": args.name, "arch": ARCH, "size": args.size, "crop": crop_cfg, "C": args.C, "aug": args.aug,
           "synth": args.synth, "real_defects_in_training": real_used, "threshold": 0.5,
           "train_counts": {k: kind.count(k) for k in sorted(set(kind))}, "feature_seconds": round(feat_s, 1),
           "made_by": "tools/brick_classifier_fm.py"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"[fm] {args.name}: {len(X)} training crops {cfg['train_counts']}, features {feat_s:.0f}s -> {out}", flush=True)


# ---------------------------------------------------------------- run time
@dataclass
class FMInspection:
    """The fields of orchestrator.Inspection, so the cell could use either."""
    verdict: str
    score: float | None
    uncertain: bool
    brick_fraction: float
    frames: int
    infer_ms: float
    low: float | None = None
    high: float | None = None
    model: str = ""
    crop: np.ndarray | None = field(default=None, repr=False)
    frame_scores: list[float] = field(default_factory=list)
    view_scores: dict = field(default_factory=dict)


class BrickClassifierFM:
    """Full inspection frame(s) -> good / defect. Same views, exposure, crops, fill and empty check as
    inspection.MultiViewScorer for the b2all fill models; score = median over frames of (max over views P(defect))."""

    def __init__(self, name: str, threshold: float | None = None, band: float = 0.15, device: str = "GPU"):
        self.dir = OUT / name
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.crop_cfg = self.cfg["crop"]
        self.views = {"frame": self.crop_cfg["frame"], "views": self.crop_cfg["views"]}
        self.empty = {n: EmptyCheck.load(self.dir / Path(r).name) for n, r in self.crop_cfg["empty_refs"].items()}
        self.fx = Features(self.cfg["size"], self.dir / "backbone.pth", device=device, xml=self.dir / "backbone.xml")
        h = np.load(self.dir / "head.npz")
        self.mean, self.scale, self.coef, self.intercept = h["mean"], h["scale"], h["coef"], float(h["intercept"][0])
        self.low = float(threshold if threshold is not None else self.cfg["threshold"])
        self.high = min(self.low + band, 0.99)
        self.name = name

    def crops(self, img_bgr: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for n, v in view_crops(img_bgr, self.views).items():
            if self.crop_cfg.get("exposure"):
                v = self.empty[n].normalise(v)
            out[n] = prep_view(v, self.empty[n], self.crop_cfg)[0]
        return out

    def proba(self, crops: list[np.ndarray]) -> np.ndarray:
        z = (self.fx(crops) - self.mean) / self.scale
        return 1 / (1 + np.exp(-(z @ self.coef + self.intercept)))

    def spot(self, img_bgr: np.ndarray) -> float:
        """MultiViewScorer.spot for a fill model: colour-hull fraction of the (exposure-normalised) direct view."""
        e = self.empty["direct"]
        v = view_crops(img_bgr, self.views)["direct"]
        if self.crop_cfg.get("exposure"):
            v = e.normalise(v)
        blob = e._blob(v)
        hull = None if blob is None else e.colour_hull(v, blob[0] == blob[1])
        return 0.0 if hull is None else float(hull.mean())

    def score(self, img_bgr: np.ndarray) -> tuple[float, dict[str, float]]:
        c = self.crops(img_bgr)
        p = self.proba(list(c.values()))
        per = {n: float(x) for n, x in zip(c, p, strict=True)}
        return max(per.values()), per

    def inspect(self, frames: list[np.ndarray]) -> FMInspection:
        fracs = [self.spot(f) for f in frames]
        frac = float(np.median(fracs))
        if frac <= self.empty["direct"].frac_thr:
            return FMInspection("empty", None, False, round(frac, 5), len(frames), 0.0, self.low, self.high, self.name)
        t = time.perf_counter()
        res = [self.score(f) for f in frames]
        ms = (time.perf_counter() - t) * 1000 / len(frames)
        scores = [s for s, _ in res]
        score = float(np.median(scores))
        mid = int(np.argsort(scores)[len(scores) // 2])
        montage = np.hstack([cv2.resize(c, (256, 256)) for c in self.crops(frames[mid]).values()])
        return FMInspection("good" if score < self.low else "defect", round(score, 4), self.low <= score < self.high,
                            round(frac, 5), len(frames), round(ms, 1), self.low, self.high, self.name, montage,
                            [round(s, 4) for s in scores], {k: round(v, 4) for k, v in res[mid][1].items()})


# ---------------------------------------------------------------- OpenVINO
def cmd_export(args) -> None:
    """backbone.pth -> backbone.xml/bin (OpenVINO IR, FP16 weights, static 1x3xSxS, output = all tokens); feature parity
    against PyTorch on real crops, and per-view latency on GPU / NPU / CPU -> config.json "openvino"."""
    import openvino as ov
    import timm
    import torch

    d = OUT / args.name
    cfg = json.loads((d / "config.json").read_text())
    s = cfg["size"]
    net = timm.create_model(ARCH, pretrained=False, img_size=s, num_classes=0)
    net.load_state_dict(torch.load(d / "backbone.pth", map_location="cpu"))
    net.eval()

    class Tokens(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):
            return self.n.forward_features(x)

    with torch.no_grad():
        ovm = ov.convert_model(Tokens(net), example_input=torch.zeros(1, 3, s, s), input=[1, 3, s, s])
    ov.save_model(ovm, str(d / "backbone.xml"), compress_to_fp16=True)
    rows = [r for r in split_rows() if r["part"] in ("test_good", "defect")][::6]
    crops = [cv2.imread(str(crop_path(r, v))) for r in rows for v in ("direct", "left", "right")]
    ref = Features(s, d / "backbone.pth")(crops)
    info = {"xml": "backbone.xml", "input": [1, 3, s, s], "fp16_weights": True, "parity_crops": len(crops)}
    for dev in args.devices:
        try:
            fx = Features(s, device=dev, xml=d / "backbone.xml")
            f = fx(crops)
            cos = (f * ref).sum(axis=1) / (np.linalg.norm(f, axis=1) * np.linalg.norm(ref, axis=1))
            fx.ms.clear()
            fx([crops[0]] * 40)
            ms = fx.ms[5:]
            info[dev] = {"ok": True, "median_ms_per_view": round(float(np.median(ms)), 2), "p90_ms": round(float(np.percentile(ms, 90)), 2),
                         "min_cosine_vs_torch": round(float(cos.min()), 5)}
        except Exception as e:  # noqa: BLE001 - report any compile / run failure per device
            info[dev] = {"ok": False, "error": str(e).splitlines()[0][:200]}
        print(f"[fm] {dev}: {info[dev]}", flush=True)
    cfg["openvino"] = info
    (d / "config.json").write_text(json.dumps(cfg, indent=1))


# ---------------------------------------------------------------- evaluation
def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def cmd_eval(args) -> None:
    """Photo level, the exact run-time pipeline from the full b2all photos (crops made by BrickClassifierFM)."""
    clf = BrickClassifierFM(args.name, device=args.device)
    rows = [r for r in split_rows() if r["part"] in ("calib_good", "test_good", "defect")]
    used = set(clf.cfg.get("real_defects_in_training", []))
    out = []
    t0 = time.time()
    for r in rows:
        img = cv2.imread(str(PHOTOS / r["class"] / r["file"]))
        s, per = clf.score(img)
        out.append({**r, "score": round(s, 4), "views": {k: round(v, 4) for k, v in per.items()},
                    "in_training": r["file"] in used})
    ms = (time.time() - t0) * 1000 / len(rows)
    g = np.array([r["score"] for r in out if r["part"] != "defect"])
    cal = np.array([r["score"] for r in out if r["part"] == "calib_good"])
    d_all = [r for r in out if r["part"] == "defect"]
    d = np.array([r["score"] for r in d_all if not r["in_training"]])
    res = {"name": args.name, "n_goods_heldout": len(g), "n_defects_eval": len(d), "auroc_heldout": round(auroc(g, d), 4),
           "ms_per_photo_3_views_cpu": round(ms, 1)}
    for thr_name, thr in (("0.5", 0.5), ("max_calib_good", float(cal.max()))):
        res[f"thr_{thr_name}"] = {"threshold": round(thr, 4), "defects_caught": f"{int((d >= thr).sum())}/{len(d)}",
                                  "goods_flagged": f"{int((g >= thr).sum())}/{len(g)}",
                                  "test_goods_flagged": f"{int((np.array([r['score'] for r in out if r['part'] == 'test_good']) >= thr).sum())}"
                                                        f"/{sum(r['part'] == 'test_good' for r in out)}"}
    per_type = {}
    for r in d_all:
        if not r["in_training"]:
            per_type.setdefault(r.get("type", "?"), []).append(r["score"])
    res["defect_scores_by_type"] = {k: [round(x, 3) for x in v] for k, v in per_type.items()}
    res["good_scores_top5"] = sorted(g.round(3).tolist())[-5:]
    res["rows"] = out
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


def tile_fill(t: np.ndarray) -> np.ndarray:
    """/tmp/live_eval.py's approximation for cycles saved only as the old 256 px montage: colour-only hull, grey fill."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(t, (5, 5), 0), cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32)
    cand = s >= 80
    if cand.sum() < 200:
        return t
    col = (cand & (s >= 0.6 * np.percentile(s[cand], 90))).astype(np.uint8)
    col = cv2.morphologyEx(col, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(col)
    if n <= 1:
        return t
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= max(30, 0.25 * big)]
    m = np.zeros(t.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(cv2.findNonZero(np.isin(lab, keep).astype(np.uint8))), 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    out = t.copy()
    out[~m] = GREY
    return out


def cmd_live(args) -> None:
    """Tonight's live cycles: full frame -> the exact pipeline; montage only -> tile_fill (as /tmp/live_eval.py)."""
    clf = BrickClassifierFM(args.name, threshold=args.threshold, device=args.device)
    truth = {}
    tp = ROOT / "results" / "truth.csv"
    if tp.exists():
        for line in tp.read_text().splitlines()[1:]:
            run, cyc, lab = line.split(",")
            truth[(run, int(cyc))] = "clean" if lab.strip() == "good" else "defect"
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "cell_runs" / "2026111[45]-*" / "cycle_*_inspect.jpg"))):
        run, ci = Path(f).parent.name, int(Path(f).name.split("_")[1])
        extra = {(a.split(":")[0], int(a.split(":")[1])) for a in args.defect}
        label = "defect" if (run, ci) in LIVE_DEFECT | extra else truth.get((run, ci), "clean")
        full = f.replace("_inspect.jpg", "_frame.jpg")
        if os.path.exists(full):
            s, per = clf.score(cv2.imread(full))
            how = "frame"
        else:
            m = cv2.imread(f)
            lab = np.zeros(m.shape[:2], np.uint8)  # the verdict text drawn into the direct tile: inpaint it
            roi = m[:60, :256].astype(np.int16)
            for c in ((60, 180, 60), (40, 60, 220)):
                lab[:60, :256] |= (np.abs(roi - np.array(c)).max(axis=2) < 60).astype(np.uint8)
            m = cv2.inpaint(m, cv2.dilate(lab, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
            tiles = [tile_fill(m[:, i * 256:(i + 1) * 256]) for i in range(3)]
            p = clf.proba(tiles)
            per = dict(zip(("direct", "left", "right"), (float(x) for x in p), strict=True))
            s, how = max(per.values()), "montage"
        rows.append({"run": run, "cycle": ci, "label": label, "score": round(s, 4), "how": how,
                     "verdict": "good" if s < clf.low else "defect", "views": {k: round(v, 3) for k, v in per.items()}})
    cl = [r for r in rows if r["label"] == "clean"]
    df = [r for r in rows if r["label"] == "defect"]
    res = {"name": args.name, "threshold": clf.low, "clean_called_good": f"{sum(r['score'] < clf.low for r in cl)}/{len(cl)}",
           "defects_caught": f"{sum(r['score'] >= clf.low for r in df)}/{len(df)}", "rows": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.name}_live.json").write_text(json.dumps(res, indent=1))
    for r in rows:
        print(f"  {r['run']} c{r['cycle']} {r['label']:6s} {r['how']:7s} {r['score']:.3f} {r['verdict']:6s} {r['views']}")
    print(f"{args.name} thr {clf.low}: clean called good {res['clean_called_good']}, defects caught {res['defects_caught']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpus", type=int, default=4)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--name", required=True)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--aug", type=int, default=1, help="photometric copies per good crop (labelled good)")
    p.add_argument("--synth", type=int, default=2, help="synthetic marker defects per good crop")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--real-defects", choices=["none", "half"], default="none")
    p = sub.add_parser("eval")
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="GPU", help="OpenVINO device of the backbone (or torch)")
    p = sub.add_parser("live")
    p.add_argument("--device", default="GPU", help="OpenVINO device of the backbone (or torch)")
    p.add_argument("--name", required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--defect", nargs="*", default=[], help="more defect cycles RUN:CYCLE (e.g. labelled by eye)")
    p = sub.add_parser("export")
    p.add_argument("--name", required=True)
    p.add_argument("--devices", nargs="+", default=["GPU", "NPU", "CPU"])
    args = ap.parse_args()
    cell_quiet()
    limit_cpu(args.cpus)
    {"train": cmd_train, "eval": cmd_eval, "live": cmd_live, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
