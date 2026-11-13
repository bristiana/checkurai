"""Run the exported Anomalib model with OpenVINO on CPU, GPU or NPU.

Library (for the closed loop):
    from anomaly.detect import Detector
    det = Detector("models/anomaly/padim", device="NPU")
    r = det(frame_bgr)        # Result(score=0.83, defect=True, ms=4.1)

CLI:
    python anomaly/detect.py models/anomaly/padim img1.png img2.png --device GPU
    python anomaly/detect.py models/anomaly/padim --cam inspection --device NPU   # live, Ctrl+C to stop
    python anomaly/detect.py models/anomaly/padim --bench                         # CPU vs GPU vs NPU table
    python anomaly/detect.py models/anomaly/padim --bench --int8
"""
import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class Result:
    score: float
    defect: bool
    ms: float


class Detector:
    def __init__(self, model_dir: str | Path, device: str = "CPU", int8: bool = False, threshold: float | None = None):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text())
        xml = self.meta["exports"]["int8" if int8 else "fp32"]
        self.device = device
        self.model = ov.Core().compile_model(xml, device, {"PERFORMANCE_HINT": "LATENCY"})
        _, _, self.h, self.w = self.model.inputs[0].shape
        self.roi = self.meta.get("roi")
        self.threshold = threshold  # None = use the label baked into the export (normalized score >= 0.5)
        names = [o.get_any_name() for o in self.model.outputs]
        self._score = next(i for i, n in enumerate(names) if "score" in n)
        self._label = next((i for i, n in enumerate(names) if "label" in n), None)

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        if self.roi:
            x, y, w, h = self.roi
            bgr = bgr[y:y + h, x:x + w]
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.w, self.h)), cv2.COLOR_BGR2RGB)
        return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    def __call__(self, bgr: np.ndarray) -> Result:
        x = self.preprocess(bgr)
        t0 = time.perf_counter()
        out = self.model(x)
        ms = (time.perf_counter() - t0) * 1000
        score = float(np.asarray(out[self._score]).ravel()[0])
        if self.threshold is not None or self._label is None:
            defect = score >= (self.threshold if self.threshold is not None else 0.5)
        else:
            defect = bool(np.asarray(out[self._label]).ravel()[0])
        return Result(round(score, 4), defect, round(ms, 2))


def bench(model_dir, images, int8, n=100):
    from bench.power import PowerMonitor

    frame = cv2.imread(str(images[0])) if images else np.random.randint(0, 255, (480, 640, 3), np.uint8)
    print(f"{'device':6s} {'p50 ms':>7s} {'p95 ms':>7s} {'fps':>6s} {'avg W':>6s} {'mJ/inf':>7s} {'pkg C':>6s}")
    rows = []
    for dev in ov.Core().available_devices:
        try:
            det = Detector(model_dir, dev, int8=int8)
        except Exception as e:  # e.g. an op the NPU does not support
            print(f"{dev:6s} failed to compile: {str(e).splitlines()[0][:80]}")
            continue
        for _ in range(10):
            det(frame)  # warm-up
        with PowerMonitor() as pm:
            t0 = time.perf_counter()
            lat = [det(frame).ms for _ in range(n)]
            wall = time.perf_counter() - t0
        s = pm.summary()
        lat.sort()
        row = {"device": dev, "p50_ms": statistics.median(lat), "p95_ms": lat[int(0.95 * n) - 1],
               "fps": round(n / wall, 1), "avg_w": s.get("avg_w"), "mj_per_inf": round(1000 * s.get("energy_j", 0) / n, 1),
               "pkg_c": s.get("x86_pkg_temp_max_c")}
        rows.append(row)
        print(f"{dev:6s} {row['p50_ms']:7.2f} {row['p95_ms']:7.2f} {row['fps']:6.1f} {row['avg_w']:6.1f} {row['mj_per_inf']:7.1f} {row['pkg_c']:6.1f}")
    out = ROOT / "logs/bench" / f"anomaly_{Path(model_dir).name}{'_int8' if int8 else ''}_{time.strftime('%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"saved {out}  (power is whole-package, includes idle baseline)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--device", default="CPU", help="CPU, GPU, NPU or AUTO")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--cam", help="camera role for live mode, e.g. inspection")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    if args.bench:
        return bench(args.model_dir, args.images, args.int8)
    det = Detector(args.model_dir, args.device, args.int8, args.threshold)
    if args.cam:
        from anomaly.capture import find_cam, open_cam
        cap = open_cam(find_cam(args.cam))
        try:
            while True:
                ok, frame = cap.read()
                if ok:
                    print(det(frame), flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
    for p in args.images:
        print(Path(p).name, det(cv2.imread(p)))


if __name__ == "__main__":
    main()
