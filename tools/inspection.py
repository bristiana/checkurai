"""Inspection helpers shared by training and the cell program.

One fixed square ROI around the inspection spot (config/inspection_roi.json), an empty-spot check
(diff against a median empty frame inside the ROI) and an OpenVINO anomaly scorer.

    from tools.inspection import load_roi, crop, EmptyCheck, AnomalyScorer

The scorer always takes the fixed ROI crop. Models trained on brick-centred crops (checkurai_meta.json
has a "crop" entry) find the brick inside the ROI themselves, so callers do not change.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ROI_PATH = ROOT / "config" / "inspection_roi.json"
EMPTY_SIZE = 160  # px; the empty check works on a small blurred copy of the ROI
MIN_BRICK_PX = 40  # px at EMPTY_SIZE; smaller blobs are noise, not a brick


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


def brick_window(roi_size: int, box: tuple[int, int, int, int] | None, side: int | None = None,
                 margin: int = 24) -> tuple[int, int, int]:
    """Square (x, y, side) in ROI pixels around the brick box, kept inside the ROI.

    side=None: the box's longer edge plus the margin on each side; otherwise a fixed side (same scale
    for every brick). No box (nothing found): the whole ROI.
    """
    if box is None:
        return 0, 0, roi_size
    x0, y0, x1, y1 = box
    s = int(side if side else max(x1 - x0, y1 - y0) + 2 * margin)
    s = min(max(s, 32), roi_size)
    x = int(np.clip(round((x0 + x1) / 2 - s / 2), 0, roi_size - s))
    y = int(np.clip(round((y0 + y1) / 2 - s / 2), 0, roi_size - s))
    return x, y, s


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

    def _changed(self, roi_bgr: np.ndarray) -> np.ndarray:
        return np.abs(_small(roi_bgr) - self.ref).max(axis=2) > self.pixel_thr

    def changed_fraction(self, roi_bgr: np.ndarray) -> float:
        return float(self._changed(roi_bgr).mean())

    def brick_present(self, roi_bgr: np.ndarray) -> bool:
        return self.changed_fraction(roi_bgr) > self.frac_thr

    def brick_box(self, roi_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """(x0, y0, x1, y1) of the largest changed blob in ROI pixels, or None when there is none."""
        m = cv2.morphologyEx(self._changed(roi_bgr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return None
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh, area = (int(v) for v in st[i])
        if area < MIN_BRICK_PX:
            return None
        k = roi_bgr.shape[0] / EMPTY_SIZE
        return int(bx * k), int(by * k), int(np.ceil((bx + bw) * k)), int(np.ceil((by + bh) * k))


class AnomalyScorer:
    """OpenVINO anomaly model exported from Anomalib (pre-processing inside the graph, raw scores out).

    device: "GPU" by default; if it cannot compile there it falls back to `fallback` (CPU), unless
    fallback=None. Call it with the fixed ROI crop.
    """

    def __init__(self, xml: Path, device: str = "GPU", image_size: int | None = None, fallback: str | None = "CPU"):
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(xml))
        meta_path = Path(xml).parent / "checkurai_meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.size = image_size or self.meta.get("image_size", 256)
        model.reshape({model.inputs[0].any_name: [1, 3, self.size, self.size]})  # static shape (needed for the NPU)
        try:
            self.compiled = core.compile_model(model, device)
        except Exception as e:  # noqa: BLE001 - any compile failure
            if not fallback or fallback == device:
                raise
            print(f"[inspection] {device} failed ({str(e).splitlines()[0][:120]}), using {fallback}", file=sys.stderr)
            device = fallback
            self.compiled = core.compile_model(model, device)
        self.device = device
        self.req = self.compiled.create_infer_request()
        self.out_names = [o.any_name for o in self.compiled.outputs]
        self.crop_cfg = self.meta.get("crop")  # None: the whole ROI goes into the model
        self.empty = EmptyCheck.load(Path(xml).parent / self.crop_cfg["empty_ref"]) if self.crop_cfg else None
        self.last_window: tuple[int, int, int] | None = None  # (x, y, side) of the last input, ROI pixels

    def window(self, roi_bgr: np.ndarray) -> tuple[int, int, int]:
        """The part of the ROI the model looks at: the whole ROI, or a square around the brick."""
        if not self.crop_cfg:
            return 0, 0, roi_bgr.shape[0]
        return brick_window(roi_bgr.shape[0], self.empty.brick_box(roi_bgr), self.crop_cfg.get("side"),
                            self.crop_cfg.get("margin", 24))

    def prepare(self, roi_bgr: np.ndarray) -> np.ndarray:
        x, y, s = self.last_window = self.window(roi_bgr)
        rgb = cv2.cvtColor(cv2.resize(roi_bgr[y:y + s, x:x + s], (self.size, self.size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
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
