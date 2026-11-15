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
SAT_MIN, SAT_GAIN = 80, 30  # tight outline: a brick pixel is saturated and clearly more saturated than the empty paper
SAT_REL = 0.6  # ... and at least this fraction of the brick's own saturation (90th percentile)


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


def brick_window(roi_size: int | tuple[int, int], box: tuple[int, int, int, int] | None, side: int | None = None,
                 margin: int = 24) -> tuple[int, int, int]:
    """Square (x, y, side) in ROI pixels around the brick box, kept inside the ROI.

    roi_size: the side of a square ROI, or (width, height) of a rectangular one (a mirror-booth view).
    side=None: the box's longer edge plus the margin on each side; otherwise a fixed side (same scale
    for every brick). No box (nothing found): the whole square ROI, or the centred square of a rectangle.
    """
    w, h = (roi_size, roi_size) if isinstance(roi_size, int) else roi_size
    if box is None:
        s = min(w, h)
        return (w - s) // 2, (h - s) // 2, s
    x0, y0, x1, y1 = box
    s = int(side if side else max(x1 - x0, y1 - y0) + 2 * margin)
    s = min(max(s, 32), w, h)
    x = int(np.clip(round((x0 + x1) / 2 - s / 2), 0, w - s))
    y = int(np.clip(round((y0 + y1) / 2 - s / 2), 0, h - s))
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

    def gain(self, roi_bgr: np.ndarray, lo: float = 0.6, hi: float = 1.6) -> np.ndarray:
        """Per-channel gain that brings this ROI's background back to the empty reference's exposure (camera
        auto-exposure / white balance drift). Median of reference / ROI over all pixels (the brick is the
        minority), then again over the pixels that match the reference after that first gain (background only)."""
        s = _small(roi_bgr) + 1.0
        r = (self.ref + 1.0) / s
        g = np.median(r.reshape(-1, 3), axis=0)
        bg = (np.abs(s * g - self.ref - 1.0).max(axis=2) < self.pixel_thr)
        if bg.sum() > 0.2 * bg.size:
            g = np.median(r[bg], axis=0)
        return np.clip(g, lo, hi).astype(np.float32)

    def normalise(self, roi_bgr: np.ndarray, base=None) -> np.ndarray:
        """The ROI with gain() applied (uint8); with `base` (the median gain of the training photos) it is brought
        to the training photos' exposure instead of the empty reference's."""
        g = self.gain(roi_bgr)
        if base is not None:
            g = g / np.asarray(base, np.float32)
        return np.clip(roi_bgr.astype(np.float32) * g, 0, 255).astype(np.uint8)

    def changed_fraction(self, roi_bgr: np.ndarray) -> float:
        return float(self._changed(roi_bgr).mean())

    def brick_present(self, roi_bgr: np.ndarray) -> bool:
        return self.changed_fraction(roi_bgr) > self.frac_thr

    def _blob(self, roi_bgr: np.ndarray) -> tuple[np.ndarray, int, np.ndarray] | None:
        """(labels, index, stats) of the largest changed blob at EMPTY_SIZE, or None when there is none."""
        m = cv2.morphologyEx(self._changed(roi_bgr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return None
        # the brick on the spot, not clutter next to it: area weighted by closeness to the ROI centre (the views
        # are the union of the brick boxes, so the spot is near the middle); one brick alone -> the largest blob
        d = np.hypot(cen[1:, 0] - EMPTY_SIZE / 2, cen[1:, 1] - EMPTY_SIZE / 2) / (0.35 * EMPTY_SIZE)
        w = st[1:, cv2.CC_STAT_AREA] * np.exp(-d * d) * (st[1:, cv2.CC_STAT_AREA] >= MIN_BRICK_PX)
        i = 1 + int(np.argmax(w))
        if int(st[i, cv2.CC_STAT_AREA]) < MIN_BRICK_PX:
            return None
        return lab, i, st

    def brick_box(self, roi_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """(x0, y0, x1, y1) of the largest changed blob in ROI pixels, or None when there is none."""
        blob = self._blob(roi_bgr)
        if blob is None:
            return None
        bx, by, bw, bh, _ = (int(v) for v in blob[2][blob[1]])
        ky, kx = roi_bgr.shape[0] / EMPTY_SIZE, roi_bgr.shape[1] / EMPTY_SIZE  # the ROI may be a rectangle
        return int(bx * kx), int(by * ky), int(np.ceil((bx + bw) * kx)), int(np.ceil((by + bh) * ky))

    def colour_hull(self, roi_bgr: np.ndarray, region: np.ndarray) -> np.ndarray | None:
        """Convex hull (EMPTY_SIZE, float 0/1) of the coloured pixels inside `region` (EMPTY_SIZE bool): saturation
        >= SAT_MIN and >= the empty reference's + SAT_GAIN. Paper whose wrinkles or shadows moved, tape and the
        black-and-white marker cards are not coloured, so they drop out; ink on the brick is inside the hull."""
        s = cv2.cvtColor(np.clip(_small(roi_bgr), 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)[..., 1].astype(np.int16)
        r = cv2.cvtColor(np.clip(self.ref, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)[..., 1].astype(np.int16)
        cand = region & (s >= SAT_MIN) & (s - r >= SAT_GAIN)
        if cand.sum() < MIN_BRICK_PX:
            return None
        # relative to this brick's own colour: shadows and reflections on the paper are only tinted
        col = (cand & (s >= SAT_REL * np.percentile(s[cand], 90))).astype(np.uint8)
        col = cv2.morphologyEx(col, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(col)
        if n <= 1:
            return None
        big = int(st[1:, cv2.CC_STAT_AREA].max())
        keep = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= max(8, 0.25 * big)]  # a brick split by a line
        if sum(int(st[k, cv2.CC_STAT_AREA]) for k in keep) < MIN_BRICK_PX:
            return None
        pts = cv2.findNonZero(np.isin(lab, keep).astype(np.uint8))
        out = np.zeros((EMPTY_SIZE, EMPTY_SIZE), np.float32)
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1.0)
        return out

    def brick_mask(self, roi_bgr: np.ndarray, dilate: int = 0, tight: bool = False) -> np.ndarray | None:
        """Brick outline in ROI pixels (bool, ROI shape): the largest changed blob with its holes filled, grown by
        `dilate` ROI px so defects on the brick's edge still count. None when there is no brick. tight: only the
        convex hull of the coloured pixels in that blob (colour_hull; the blob itself if there are none), because
        under changed light or moved paper the changed blob also takes in background and marker cards."""
        blob = self._blob(roi_bgr)
        if blob is None:
            return None
        cnts, _ = cv2.findContours((blob[0] == blob[1]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros((EMPTY_SIZE, EMPTY_SIZE), np.float32)
        cv2.drawContours(filled, cnts, -1, 1.0, cv2.FILLED)
        if tight:
            hull = self.colour_hull(roi_bgr, filled > 0)
            if hull is not None:
                filled = hull
        h, w = roi_bgr.shape[:2]
        m = (cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
        if dilate > 0:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
        return m > 0


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
        has_ref = self.crop_cfg and "empty_ref" in self.crop_cfg  # mirror-booth models keep one ref per view
        self.empty = EmptyCheck.load(Path(xml).parent / self.crop_cfg["empty_ref"]) if has_ref else None
        self.last_window: tuple[int, int, int] | None = None  # (x, y, side) of the last input, ROI pixels

    def window(self, roi_bgr: np.ndarray) -> tuple[int, int, int]:
        """The part of the ROI the model looks at: the whole ROI, or a square around the brick."""
        if not self.empty:
            return 0, 0, roi_bgr.shape[0]
        return brick_window(roi_bgr.shape[0], self.empty.brick_box(roi_bgr), self.crop_cfg.get("side"),
                            self.crop_cfg.get("margin", 24))

    def prepare(self, roi_bgr: np.ndarray) -> np.ndarray:
        x, y, s = self.last_window = self.window(roi_bgr)
        return self.to_input(roi_bgr[y:y + s, x:x + s])

    def to_input(self, square_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(square_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    def __call__(self, roi_bgr: np.ndarray) -> tuple[float, np.ndarray | None, float]:
        """Return (raw anomaly score, anomaly map or None, inference ms)."""
        return self.infer(self.prepare(roi_bgr))

    def infer(self, x: np.ndarray) -> tuple[float, np.ndarray | None, float]:
        """Run the model on one prepared input (to_input); (raw score, anomaly map or None, ms)."""
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


# ---------------------------------------------------------------- mirror booth (direct view + two mirrors)
VIEWS_PATH = ROOT / "config" / "inspection_views.json"


def load_views(path: Path = VIEWS_PATH) -> dict:
    """{"frame": [w, h], "views": {name: [x0, y0, x1, y1]}} in full-frame pixels."""
    return json.loads(Path(path).read_text())


def view_crops(img_bgr: np.ndarray, views: dict) -> dict[str, np.ndarray]:
    """One rectangle per view (direct, left mirror, right mirror) from a full-resolution inspection frame."""
    h, w = img_bgr.shape[:2]
    if (w, h) != tuple(views["frame"]):
        raise ValueError(f"frame is {w}x{h}, views were made for {views['frame'][0]}x{views['frame'][1]}")
    return {n: img_bgr[y0:y1, x0:x1] for n, (x0, y0, x1, y1) in views["views"].items()}


def prep_view(v: np.ndarray, empty: "EmptyCheck", cfg: dict) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray | None]:
    """One view rectangle -> (model input square, (x, y, side) in the view, fill outline inside the square or None).

    cfg is the model's crop config (checkurai_meta.json "crop" = the crop set's crop.json): side, margin and
    fill = {"dilate": px, "value": grey} or absent. Without fill: the square around the changed blob (brick_box),
    as before. With fill: the brick is the tight outline (EmptyCheck.brick_mask tight=True, grown by fill["dilate"]
    view px), the square goes around that outline, and every pixel outside it becomes the constant grey, so the
    model never sees paper, marker cards, shadows or the mirror rims (prepare-views --fill makes the training
    crops the same way). Exposure normalisation, if any, is applied to `v` before this (MultiViewScorer.crops)."""
    fill = cfg.get("fill")
    mk = None
    if fill:
        mk = empty.brick_mask(v, int(fill.get("dilate", 8)), tight=True)
        box = None
        if mk is not None:
            ys, xs = np.nonzero(mk)
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    else:
        box = empty.brick_box(v)
    x, y, s = brick_window((v.shape[1], v.shape[0]), box, cfg.get("side"), cfg.get("margin", 24))
    sq = v[y:y + s, x:x + s]
    if mk is not None:
        mk = mk[y:y + s, x:x + s]
        sq = sq.copy()
        sq[~mk] = int(fill.get("value", 128))
    return sq, (x, y, s), mk


class MultiViewScorer:
    """Anomaly score of a brick in the mirror booth (direct view + two mirrors); the brick's score is its worst view.

    `path` is a model folder (or its model.xml) whose checkurai_meta.json "crop" is either
      {"mode": "views", ...}    one model for all three views, or
      {"mode": "viewset", "models": {view: xml path relative to the folder}, ...}   one model per view.
    Either way the meta carries the view rectangles and one empty-spot reference per view; in each view the
    brick is the largest blob that differs from that reference, and a square around it (box + margin) goes
    into that view's model. norm = {view: [median, max]} of the calibration good scores (from eval-views):
    each view's score becomes (s - median) / (max - median), so the views are compared on one scale.
    mask = None: a view's score is the model's image score. mask = {"dilate": px, "pct": p} (or True for the
    defaults): the score is the p-th percentile (100 = max) of the anomaly map inside the brick outline (the
    blob that differs from the view's empty reference, holes filled, grown by `dilate` view px), so the
    background, plate and mirror edges cannot score; "tight": True keeps only the convex hull of the coloured
    pixels of that blob (EmptyCheck.colour_hull). The norm must come from the same scoring (eval-views --mask).
    exposure = True: each view is first brought back to its empty reference's exposure (EmptyCheck.gain), for the
    spot check, the brick outline and the model input alike (camera auto-exposure drift); exposure = {view: [b, g, r]}
    (eval-views --exposure): to the training photos' exposure instead (that view's gain divided by their median gain).
    Call it with the full inspection frame.
    """

    MASK_DEFAULTS = {"dilate": 12, "pct": 100.0, "tight": False}

    def __init__(self, path: Path, device: str = "GPU", fallback: str | None = "CPU", norm: dict | None = None,
                 mask: dict | bool | None = None, exposure: bool | dict = False):
        path = Path(path)
        mdir = path if path.is_dir() else path.parent
        self.meta = json.loads((mdir / "checkurai_meta.json").read_text())
        cfg = self.cfg = self.meta.get("crop") or {}
        if cfg.get("mode") == "viewset":
            self.models = {n: AnomalyScorer((mdir / rel).resolve(), device=device, fallback=fallback)
                           for n, rel in cfg["models"].items()}
        elif cfg.get("mode") == "views":
            shared = AnomalyScorer(path if path.suffix == ".xml" else next(mdir.glob("*.xml")), device=device, fallback=fallback)
            self.models = {n: shared for n in cfg["views"]}
        else:
            raise ValueError(f"{mdir} is not a mirror-booth (views / viewset) model")
        first = next(iter(self.models.values()))
        self.device, self.size = first.device, first.size
        self.views = {"frame": cfg["frame"], "views": cfg["views"]}
        self.empty = {n: EmptyCheck.load(mdir / ref) for n, ref in cfg["empty_refs"].items()}
        self.norm = norm
        self.mask = None if not mask else {**self.MASK_DEFAULTS, **(mask if isinstance(mask, dict) else {})}
        self.exposure = exposure or cfg.get("exposure") or False  # a fill model is trained on normalised crops
        self.last: dict[str, dict] = {}  # per view: score, window (x, y, side in view pixels), ms (+ model score, mask)

    def spot(self, img_bgr: np.ndarray) -> float:
        """Changed fraction of the direct view against its empty reference (brick present if > frac_thr). For a fill
        model: the fraction of the direct view covered by the brick's colour hull (0 when the changed blob has no
        coloured pixels), so moved paper, shadows or a light change on an empty spot do not count as a brick."""
        e, v = self.empty["direct"], self.crops(img_bgr, ("direct",))["direct"]
        if not self.cfg.get("fill"):
            return e.changed_fraction(v)
        blob = e._blob(v)
        hull = None if blob is None else e.colour_hull(v, blob[0] == blob[1])
        return 0.0 if hull is None else float(hull.mean())

    def crops(self, img_bgr: np.ndarray, names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        """The view rectangles (only `names` if given), exposure-normalised when self.exposure."""
        out = {n: v for n, v in view_crops(img_bgr, self.views).items() if names is None or n in names}
        if not self.exposure:
            return out
        base = self.exposure if isinstance(self.exposure, dict) else {}
        return {n: self.empty[n].normalise(v, base.get(n)) for n, v in out.items()}

    def windows(self, img_bgr: np.ndarray, with_mask: bool = False) -> dict[str, tuple]:
        """{view: (model input square, (x, y, side))}; with_mask adds the brick outline inside the square (or None)."""
        out = {}
        for n, v in self.crops(img_bgr).items():
            sq, (x, y, s), _ = prep_view(v, self.empty[n], self.cfg)
            out[n] = (sq, (x, y, s))
            if with_mask:
                mk = self.empty[n].brick_mask(v, int(self.mask["dilate"]), bool(self.mask.get("tight")))
                out[n] += (None if mk is None else mk[y:y + s, x:x + s],)
        return out

    def masked_score(self, amap: np.ndarray, mask: np.ndarray | None) -> float:
        """Percentile (max at 100) of the anomaly map inside the brick outline; the whole map when there is none."""
        amap = np.asarray(amap, np.float32)
        if mask is not None:
            mask = cv2.resize(mask.astype(np.uint8), (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        vals = amap[mask] if mask is not None and mask.any() else amap.reshape(-1)
        pct = float(self.mask["pct"])
        return float(vals.max()) if pct >= 100 else float(np.percentile(vals, pct))

    def __call__(self, img_bgr: np.ndarray) -> tuple[float, dict[str, float], float]:
        """(score = max over views of the normalised scores, {view: raw score}, total inference ms)."""
        self.last, total = {}, 0.0
        for n, item in self.windows(img_bgr, with_mask=self.mask is not None).items():
            sq, win = item[:2]
            m = self.models[n]
            s, amap, ms = m.infer(m.to_input(sq))
            self.last[n] = {"score": s, "window": win, "ms": ms}
            if self.mask is not None and amap is not None:
                self.last[n].update(model_score=s, score=self.masked_score(amap, item[2]),
                                    mask_px=None if item[2] is None else int(item[2].sum()))
            total += ms
        per = {n: r["score"] for n, r in self.last.items()}
        return self.combine(per), per, total

    def combine(self, per: dict[str, float]) -> float:
        if not self.norm:
            return max(per.values())
        return max((s - self.norm[n][0]) / max(self.norm[n][1] - self.norm[n][0], 1e-6) for n, s in per.items())

    def montage(self, img_bgr: np.ndarray, side: int = 256) -> np.ndarray:
        """The three model inputs side by side (for the cycle log)."""
        return np.hstack([cv2.resize(sq, (side, side), interpolation=cv2.INTER_AREA) for sq, _ in self.windows(img_bgr).values()])
