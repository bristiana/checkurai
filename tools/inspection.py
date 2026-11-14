"""Inspection helpers shared by training and the cell program.

One fixed square ROI around the inspection spot (config/inspection_roi.json), an empty-spot check
(diff against a median empty frame inside the ROI) and an OpenVINO anomaly scorer.

    from tools.inspection import load_roi, crop, EmptyCheck, AnomalyScorer
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ROI_PATH = ROOT / "config" / "inspection_roi.json"
EMPTY_SIZE = 160  # px; the empty check works on a small blurred copy of the ROI


def load_roi(path: Path = ROI_PATH) -> dict:
    return json.loads(Path(path).read_text())


def crop(img_bgr: np.ndarray, roi: dict) -> np.ndarray:
    """Square ROI crop from a full-resolution inspection frame."""
    x, y, s = roi["x"], roi["y"], roi["size"]
    h, w = img_bgr.shape[:2]
    if (w, h) != tuple(roi["frame"]):
        raise ValueError(f"frame is {w}x{h}, ROI was made for {roi['frame'][0]}x{roi['frame'][1]}")
    return img_bgr[y:y + s, x:x + s]


def _small(roi_bgr: np.ndarray) -> np.ndarray:
    im = cv2.resize(roi_bgr, (EMPTY_SIZE, EMPTY_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(im, (5, 5), 0).astype(np.float32)


class EmptyCheck:
    """Is there a brick on the spot? Fraction of ROI pixels that differ from the median empty ROI."""

    def __init__(self, reference: np.ndarray, pixel_thr: float = 35.0, frac_thr: float = 0.02):
        self.ref = _small(reference)
        self.pixel_thr = pixel_thr
        self.frac_thr = frac_thr

    @classmethod
    def from_crops(cls, crops: list[np.ndarray], **kw) -> "EmptyCheck":
        return cls(np.median(np.stack(crops), axis=0).astype(np.uint8), **kw)

    @classmethod
    def load(cls, path: Path) -> "EmptyCheck":
        meta = json.loads(Path(path).with_suffix(".json").read_text())
        return cls(cv2.imread(str(path)), pixel_thr=meta["pixel_thr"], frac_thr=meta["frac_thr"])

    def save(self, path: Path) -> None:
        cv2.imwrite(str(path), cv2.resize(self.ref.astype(np.uint8), (EMPTY_SIZE, EMPTY_SIZE)))
        Path(path).with_suffix(".json").write_text(json.dumps({"pixel_thr": self.pixel_thr, "frac_thr": self.frac_thr}))

    def changed_fraction(self, roi_bgr: np.ndarray) -> float:
        d = np.abs(_small(roi_bgr) - self.ref).max(axis=2)
        return float((d > self.pixel_thr).mean())

    def brick_present(self, roi_bgr: np.ndarray) -> bool:
        return self.changed_fraction(roi_bgr) > self.frac_thr


class AnomalyScorer:
    """OpenVINO anomaly model exported from Anomalib (pre-processing inside the graph, raw scores out)."""

    def __init__(self, xml: Path, device: str = "GPU", image_size: int | None = None):
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(xml))
        meta_path = Path(xml).parent / "checkurai_meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.size = image_size or self.meta.get("image_size", 256)
        model.reshape({model.inputs[0].any_name: [1, 3, self.size, self.size]})  # static shape (needed for the NPU)
        self.device = device
        self.compiled = core.compile_model(model, device)
        self.req = self.compiled.create_infer_request()
        self.out_names = [o.any_name for o in self.compiled.outputs]

    def prepare(self, roi_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(roi_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    def __call__(self, roi_bgr: np.ndarray) -> tuple[float, np.ndarray | None, float]:
        """Return (raw anomaly score, anomaly map or None, inference ms)."""
        x = self.prepare(roi_bgr)
        t = time.perf_counter()
        res = self.req.infer({0: x})
        ms = (time.perf_counter() - t) * 1000
        score, amap = None, None
        for out, val in zip(self.compiled.outputs, res.values(), strict=False):
            names = out.get_names()
            if any("score" in n for n in names):
                score = float(np.asarray(val).reshape(-1)[0])
            elif any("map" in n for n in names):
                amap = np.asarray(val).squeeze()
        if score is None:  # fall back to the max of the map
            score = float(np.max(amap))
        return score, amap, ms
