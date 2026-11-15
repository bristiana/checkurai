#!/usr/bin/env python3
"""Supervised brick inspection: P(defect) per view from a small CNN, same view crops and empty check as the
anomaly models, OpenVINO on the GPU / NPU / CPU.

The model (models/classifier/<name>/) is an ImageNet resnet18 fine-tuned on the background-neutral per-view crops
(data/inspection_crops_views_b2all, see its PREPROC.md): real clean crops are "good", the same crops with synthetic
marker strokes (dots, lines, scribbles; black, grey, dark blue) are "defect", with brightness / white-balance /
specular-highlight augmentation so light and gloss do not read as defects. Real defect photos were only used to
pick the threshold (meta.json "threshold": at most one held-out good false alarm).

BrickClassifierInspector has AnomalibInspector's interface (tools/orchestrator.py): spot(frames) -> (present, frac),
inspect(frames) -> Inspection(verdict, score = P(defect) max over views, median over frames, uncertain, ...).
Score < low: good; low <= score < high: defect, uncertain; >= high: defect.

  python tools/brick_classifier.py export NAME          # model.pt -> OpenVINO IR (model.xml) next to it
  python tools/brick_classifier.py bench NAME --device NPU
  python tools/brick_classifier.py score NAME photo.jpg [...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import inspection as insp  # noqa: E402

MODELS = ROOT / "models" / "classifier"
VIEWS = ("direct", "left", "right")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class ViewCropper(insp.MultiViewScorer):
    """MultiViewScorer's crop half only (view rects, exposure normalisation, tight outline + grey fill, spot check),
    without an anomaly model."""

    def __init__(self, crop_cfg: dict, ref_dir: Path):  # noqa: D107  (no super().__init__: no model to load)
        self.cfg = crop_cfg
        self.views = {"frame": crop_cfg["frame"], "views": crop_cfg["views"]}
        self.empty = {n: insp.EmptyCheck.load(Path(ref_dir) / ref) for n, ref in crop_cfg["empty_refs"].items()}
        self.exposure = crop_cfg.get("exposure") or False
        self.mask, self.norm, self.last = None, None, {}


def preprocess(crops: list[np.ndarray], size: int) -> np.ndarray:
    x = np.stack([cv2.cvtColor(cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                  for c in crops]).astype(np.float32) / 255
    return ((x - MEAN) / STD).transpose(0, 3, 1, 2).copy()


class BrickClassifierInspector:
    """Drop-in for orchestrator.AnomalibInspector (same spot / inspect interface)."""

    def __init__(self, model_name: str, device: str = "GPU", force: str | None = None):
        self.dir = MODELS / model_name
        self.meta = json.loads((self.dir / "meta.json").read_text())
        crop = json.loads((self.dir / "crop.json").read_text())
        self.cropper = ViewCropper(crop, self.dir)
        self.empty = self.cropper.empty["direct"]
        self.scorer = self.cropper  # cell_live's dashboard hooks read inspector.scorer (view rects, norm)
        self.size = int(self.meta["size"])
        self.low = float(self.meta["threshold"])
        self.high = float(self.meta.get("threshold_high", self.low + 0.5 * (1 - self.low)))
        self.model_name, self.force, self.views = model_name, force, True
        self.last_views: dict[str, float] = {}
        xml = self.dir / "model.xml"
        if xml.exists():
            import openvino as ov

            core = ov.Core()
            model = core.read_model(xml)
            model.reshape([1, 3, self.size, self.size])  # static shape (the NPU needs it)
            self.device = device if device in core.available_devices or device == "CPU" else "CPU"
            try:
                self.compiled = core.compile_model(model, self.device)
            except Exception:  # noqa: BLE001  (device plugin missing or cannot compile: CPU)
                self.device = "CPU"
                self.compiled = core.compile_model(model, "CPU")
            self.torch = None
        else:  # no IR yet: torch on the CPU
            self.torch, self.device = load_torch(self.dir), "CPU(torch)"

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """P(defect) per crop."""
        x = preprocess(crops, self.size)
        if self.torch is not None:
            import torch

            with torch.no_grad():
                logit = self.torch(torch.from_numpy(x)).numpy()[:, 0]
        else:
            logit = np.concatenate([np.asarray(self.compiled(x[i:i + 1])[0]).reshape(-1) for i in range(len(x))])
        if self.meta.get("output") == "probability":  # e.g. SuperSimpleNet: the IR already ends in a sigmoid
            return logit
        return 1 / (1 + np.exp(-logit))

    def cam(self, crop: np.ndarray) -> np.ndarray:
        """Where the classifier sees the defect: class activation map of one view crop, HxW float in [0, 1] at the
        crop's size (sigmoid of the per-cell logit, 7x7 cells upsampled; the cells' mean logit is the model's
        logit). OpenVINO cam.xml (written by `export`) on the inspector's device, a few ms per crop; torch on the
        CPU if cam.xml is missing; zeros for non-resnet models."""
        if not hasattr(self, "_cam"):
            self._cam = None
            try:
                xml = self.dir / "cam.xml"
                if xml.exists():
                    import openvino as ov

                    core = ov.Core()
                    m = core.read_model(xml)
                    m.reshape([1, 3, self.size, self.size])
                    dev = self.device if self.device in ("GPU", "NPU", "CPU") else "CPU"
                    self._cam = ("ov", core.compile_model(m, dev))
                elif (self.dir / "model.pt").exists() or self.meta.get("members"):
                    self._cam = ("torch", cam_net(self.dir))
            except Exception:  # noqa: BLE001  (no map rather than a failed inspection)
                self._cam = None
        if self._cam is None:
            return np.zeros(crop.shape[:2], np.float32)
        x = preprocess([crop], self.size)
        if self._cam[0] == "ov":
            c = np.asarray(self._cam[1](x)[0])
        else:
            import torch

            with torch.no_grad():
                c = self._cam[1](torch.from_numpy(x)).numpy()
        c = 1 / (1 + np.exp(-c.reshape(c.shape[-2:])))
        return cv2.resize(c.astype(np.float32), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC).clip(0, 1)

    def spot(self, frames: list[np.ndarray]) -> tuple[bool, float]:
        frac = float(np.median([self.cropper.spot(f) for f in frames]))
        return frac > self.empty.frac_thr, frac

    def score_frame(self, frame: np.ndarray) -> tuple[float, dict[str, np.ndarray], float]:
        w = self.cropper.windows(frame)
        t = time.perf_counter()
        p = self.predict([w[n][0] for n in VIEWS])
        ms = (time.perf_counter() - t) * 1000
        self.last_views = dict(zip(VIEWS, (float(v) for v in p)))
        return float(p.max()), {n: w[n][0] for n in VIEWS}, ms

    def montage(self, crops: dict[str, np.ndarray], side: int = 256) -> np.ndarray:
        return np.hstack([cv2.resize(crops[n], (side, side), interpolation=cv2.INTER_AREA) for n in VIEWS])

    def inspect(self, frames: list[np.ndarray]):
        from tools.orchestrator import Inspection

        present, frac = self.spot(frames)
        if not present:
            return Inspection("empty", None, False, frac, len(frames), 0.0, self.low, self.high, self.model_name)
        scores, ms, crops = [], [], []
        for f in frames:
            s, c, t = self.score_frame(f)
            scores.append(s)
            ms.append(t)
            crops.append(c)
        score = float(np.median(scores))
        uncertain = self.low <= score < self.high
        verdict = "good" if score < self.low else "defect"
        if self.force:
            verdict = self.force
        mid = int(np.argsort(scores)[len(scores) // 2])
        return Inspection(verdict, round(score, 4), uncertain, round(frac, 5), len(frames), round(float(np.median(ms)), 2),
                          self.low, self.high, self.model_name, self.montage(crops[mid]),
                          [round(float(s), 4) for s in scores], frames[mid])


def load_torch(mdir: Path):
    import torch
    import torch.nn as nn
    import torchvision

    net = torchvision.models.resnet18()
    net.fc = nn.Linear(512, 1)
    net.load_state_dict(torch.load(mdir / "model.pt", map_location="cpu"))
    return net.eval()


def cam_net(mdir: Path):
    """Class activation map of a resnet18 classifier (or the mean over an ensemble's members): the fc layer applied
    to each layer4 cell (1x1 conv) -> (B, 1, 7, 7) per-cell logits whose mean is exactly the model's logit."""
    import torch
    import torch.nn.functional as F

    meta = json.loads((mdir / "meta.json").read_text())
    nets = [load_torch(MODELS / m) for m in meta["members"]] if meta.get("members") else [load_torch(mdir)]

    class CamNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.nets = torch.nn.ModuleList(nets)

        def forward(self, x):
            out = []
            for n in self.nets:
                f = n.layer4(n.layer3(n.layer2(n.layer1(n.maxpool(n.relu(n.bn1(n.conv1(x))))))))
                out.append(F.conv2d(f, n.fc.weight[:, :, None, None], n.fc.bias))
            return torch.stack(out).mean(0)

    return CamNet().eval()


def cmd_export(args) -> None:
    import openvino as ov
    import torch

    mdir = MODELS / args.name
    size = json.loads((mdir / "meta.json").read_text())["size"]
    x = torch.zeros(1, 3, size, size)
    if not args.cam_only:
        ov.save_model(ov.convert_model(load_torch(mdir), example_input=x, input=[1, 3, size, size]), mdir / "model.xml")
        print(f"OpenVINO IR -> {mdir / 'model.xml'}")
    ov.save_model(ov.convert_model(cam_net(mdir), example_input=x, input=[1, 3, size, size]), mdir / "cam.xml")
    print(f"class activation map IR -> {mdir / 'cam.xml'}")


def cmd_bench(args) -> None:
    ins = BrickClassifierInspector(args.name, args.device)
    crops = [np.full((300, 300, 3), 128, np.uint8)] * 3
    for _ in range(3):
        ins.predict(crops)
    t = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        ins.predict(crops)
        t.append((time.perf_counter() - t0) * 1000)
    print(f"{args.name} on {ins.device}: 3 views {np.median(t):.1f} ms median ({np.percentile(t, 90):.1f} p90)")
    if args.cam:
        for c in crops:
            ins.cam(c)
        t = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            for c in crops:
                ins.cam(c)
            t.append((time.perf_counter() - t0) * 1000)
        print(f"  + cam maps, 3 views: {np.median(t):.1f} ms median ({np.percentile(t, 90):.1f} p90)")


def cmd_score(args) -> None:
    ins = BrickClassifierInspector(args.name, args.device)
    for p in args.photos:
        r = ins.inspect([cv2.imread(p)])
        print(p, r.verdict, r.score, {k: round(v, 3) for k, v in ins.last_views.items()})
        if args.heat and r.crop is not None:  # crops | crops with the class activation map
            _, crops, _ = ins.score_frame(cv2.imread(p))
            tiles = []
            for n in VIEWS:
                c = cv2.resize(crops[n], (256, 256), interpolation=cv2.INTER_AREA)
                h = cv2.applyColorMap((ins.cam(c) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                tiles.append(np.vstack([c, cv2.addWeighted(c, 0.55, h, 0.45, 0)]))
            out = Path(args.heat) / f"{Path(p).parent.name}_{Path(p).stem}_cam.jpg"
            cv2.imwrite(str(out), np.hstack(tiles))
            print("  ->", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("name")
    p.add_argument("--cam-only", action="store_true", help="only cam.xml (leave a deployed model.xml untouched)")
    p = sub.add_parser("bench")
    p.add_argument("name")
    p.add_argument("--device", default="GPU")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--cam", action="store_true", help="also time the class activation maps")
    p = sub.add_parser("score")
    p.add_argument("name")
    p.add_argument("photos", nargs="+")
    p.add_argument("--device", default="CPU")
    p.add_argument("--heat", metavar="DIR", help="write <photo>_cam.jpg (view crops + class activation map) here")
    args = ap.parse_args()
    {"export": cmd_export, "bench": cmd_bench, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
