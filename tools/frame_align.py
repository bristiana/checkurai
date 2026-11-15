"""Undo small inspection-camera moves: map each full-resolution frame onto the geometry of the anomaly photos.

The anomaly model, its view rectangles (config/inspection_views.json) and its empty-spot references were all
made from photos taken with the inspection camera in one position (it stayed within ~4 px over all 144 photos).
If the camera is bumped, the brick lands elsewhere in the view crops and the empty-spot check sees "changes"
everywhere. The ArUco markers next to the stage (4 and 5) are found in the frame, and a similarity transform
(shift, turn, zoom) from their corners maps the frame back onto the photos' geometry before cropping/scoring.

    python tools/frame_align.py build          # reference = median marker corners over the anomaly photos
    python tools/frame_align.py test           # synthetic bumps on photos: scores before/after alignment
    python tools/frame_align.py live           # how far the live camera is off now (needs the camera free)

In the cell program: InspectionCamera.grab() returns aligned frames when config/cell_ref/inspection_train_markers.json
exists (tools/orchestrator.py).
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "config" / "cell_ref" / "inspection_train_markers.json"
PHOTOS = ROOT / "data" / "inspection"
_DET = None


def detect(img: np.ndarray) -> dict[int, np.ndarray]:
    global _DET
    if _DET is None:
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), p)
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c, ids, _ = _DET.detectMarkers(g)
    return {} if ids is None else {int(i): k[0].astype(np.float64) for i, k in zip(ids.ravel(), c)}


def photo_paths() -> list[Path]:
    return [f for f in sorted(PHOTOS.rglob("*.jpg")) if not ({"_unused", "_junk"} & set(f.parts))]


def build() -> dict:
    seen, size = [], None
    for f in photo_paths():
        im = cv2.imread(str(f))
        size = size or [im.shape[1], im.shape[0]]
        if [im.shape[1], im.shape[0]] == size:
            seen.append(detect(im))
    ids = sorted({i for s in seen for i in s})
    markers, spread = {}, {}
    for i in ids:
        cs = [s[i] for s in seen if i in s]
        if len(cs) >= 0.5 * len(seen):
            med = np.median(np.stack(cs), 0)
            markers[str(i)] = med.round(2).tolist()
            spread[str(i)] = round(float(np.percentile([np.linalg.norm(c.mean(0) - med.mean(0)) for c in cs], 90)), 1)
    out = {"frame": size, "markers": markers, "photos": len(seen), "spread_p90_px": spread,
           "made_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    REF.write_text(json.dumps(out, indent=1) + "\n")
    return out


def similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (shift, turn, zoom) mapping src points onto dst, as a 2x3 matrix."""
    zs, zd = src[:, 0] + 1j * src[:, 1], dst[:, 0] + 1j * dst[:, 1]
    ms, md = zs.mean(), zd.mean()
    a = np.sum(np.conj(zs - ms) * (zd - md)) / np.sum(np.abs(zs - ms) ** 2)
    t = md - a * ms
    return np.array([[a.real, -a.imag, t.real], [a.imag, a.real, t.imag]])


class FrameAligner:
    """align(frame) -> frame in the photos' geometry. Keeps the last good transform for frames where the markers
    are covered (the camera does not move between frames); `info` says what was done."""

    def __init__(self, ref_path: Path = REF, min_shift: float = 4.0, max_shift: float = 150.0, max_resid: float = 6.0):
        r = json.loads(Path(ref_path).read_text())
        self.size = tuple(r["frame"])
        self.ref = {int(i): np.array(c) for i, c in r["markers"].items()}
        self.min_shift, self.max_shift, self.max_resid = min_shift, max_shift, max_resid
        self.last: tuple[np.ndarray, float] | None = None  # (transform, monotonic time)
        self.info: dict = {}

    @classmethod
    def try_load(cls, **kw) -> "FrameAligner | None":
        return cls(**kw) if REF.exists() else None

    def fit(self, img: np.ndarray) -> dict:
        live = detect(img)
        common = sorted(set(live) & set(self.ref))
        if not common:
            return {"ok": False, "why": f"no reference marker visible (want {sorted(self.ref)}, see {sorted(live)})"}
        src = np.concatenate([live[i] for i in common])
        dst = np.concatenate([self.ref[i] for i in common])
        if len(common) >= 2:  # marker centres: far apart and steadier than single corners (marker 4's wobble)
            A = similarity(np.array([live[i].mean(0) for i in common]), np.array([self.ref[i].mean(0) for i in common]))
        else:
            A = similarity(src, dst)
        resid = float(np.linalg.norm(src @ A[:, :2].T + A[:, 2] - dst, axis=1).max())  # on the corners
        w, h = self.size
        pts = np.array([[0, 0], [w, 0], [0, h], [w, h], [w / 2, h / 2]], np.float64)
        shift = float(np.linalg.norm(pts @ A[:, :2].T + A[:, 2] - pts, axis=1).max())
        zoom = float(np.hypot(A[0, 0], A[1, 0]))
        ok = (resid <= (self.max_resid if len(common) == 1 else 2.5 * self.max_resid) and shift <= self.max_shift
              and abs(zoom - 1) < 0.06)
        return {"ok": ok, "A": A, "markers": common, "residual_px": round(resid, 2), "shift_px": round(shift, 1),
                "why": "" if ok else f"rejected (residual {resid:.1f} px, shift {shift:.0f} px, markers {common})"}

    def align(self, img: np.ndarray) -> np.ndarray:
        if (img.shape[1], img.shape[0]) != self.size:
            self.info = {"ok": False, "why": f"frame {img.shape[1]}x{img.shape[0]} != reference {self.size}"}
            return img
        f = self.fit(img)
        now = time.monotonic()
        A = None
        if f["ok"]:
            A, self.last = f["A"], (f["A"], now)
        elif "markers" not in f and self.last is not None and now - self.last[1] < 120:
            # markers covered (a hand, the arm): the camera has not moved since the last good fit seconds ago
            A, f = self.last[0], {**f, "used": "transform from the last frame with visible markers"}
        self.info = {k: v for k, v in f.items() if k != "A"}
        if A is None or (f["ok"] and f["shift_px"] < self.min_shift):
            return img
        return cv2.warpAffine(img, A, self.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def bump(img: np.ndarray, dx: float, dy: float, deg: float, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    A = cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)
    A[:, 2] += (dx, dy)
    return cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def test(model: str | None) -> None:
    sys.path.insert(0, str(ROOT))
    from tools import inspection as insp

    cfg = json.loads((ROOT / "config" / "cell.json").read_text())
    model = model or cfg["anomaly_model"]
    meta_dir = ROOT / "models" / "anomaly" / model / "weights" / "openvino"
    res = json.loads((ROOT / "results" / "anomaly" / f"{model}.json").read_text())
    sc = insp.MultiViewScorer(meta_dir, device="GPU", norm=res.get("view_norm"))
    al = FrameAligner()
    bumps = [(0, 0, 0, 1.0), (25, -15, 0, 1.0), (-40, 20, 0.8, 1.0), (15, 30, -1.5, 1.02), (60, 0, 0, 1.0)]
    rng = np.random.default_rng(0)
    paths = photo_paths()
    picks = []
    for cls in ("good", "defect", "empty"):
        ps = [p for p in paths if p.parent.name == cls]
        picks += [ps[i] for i in rng.choice(len(ps), min(3, len(ps)), replace=False)]
    print(f"model {model}; thresholds low {res['threshold']['low']:.3f} high {res['threshold']['high']:.3f}")
    print("bump (dx,dy,deg,scale): mean |score change| raw -> aligned; spot changed-fraction on empty raw -> aligned")
    base = {}
    for p in picks:
        im = cv2.imread(str(p))
        base[p] = (sc(im)[0], sc.spot(im))
    for b in bumps:
        d_raw, d_al, sp_raw, sp_al, flips_raw, flips_al = [], [], [], [], 0, 0
        for p in picks:
            im = bump(cv2.imread(str(p)), *b)
            imal = al.align(im)
            s0, f0 = base[p]
            sr, sa = sc(im)[0], sc(imal)[0]
            d_raw.append(abs(sr - s0))
            d_al.append(abs(sa - s0))
            hi = res["threshold"]["low"]
            flips_raw += (sr >= hi) != (s0 >= hi)
            flips_al += (sa >= hi) != (s0 >= hi)
            if p.parent.name == "empty":
                sp_raw.append(sc.spot(im))
                sp_al.append(sc.spot(imal))
        print(f"{b}: score change {np.mean(d_raw):.3f} -> {np.mean(d_al):.3f}; verdict flips {flips_raw} -> {flips_al} "
              f"of {len(picks)}; empty-spot change {np.mean(sp_raw) if sp_raw else 0:.3f} -> "
              f"{np.mean(sp_al) if sp_al else 0:.3f}; aligner: {al.info}")


def live() -> None:
    sys.path.insert(0, str(ROOT))
    from tools.orchestrator import InspectionCamera

    cam = InspectionCamera().start()
    try:
        fr = cam.grab(3)[-1]
    finally:
        cam.stop()
    al = FrameAligner()
    al.align(fr)
    print(f"live inspection camera vs the anomaly photos: {al.info}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        print(json.dumps(build()))
    elif cmd == "test":
        test(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "live":
        live()
