"""Train an Anomalib model on data/anomaly/{good,defect} and export it to OpenVINO.

Usage (on the Intel laptop, intel_dev_env):
    python anomaly/train.py --model padim             # ~1 min, good first baseline
    python anomaly/train.py --model patchcore --int8  # better accuracy, INT8 for the NPU

Only good/ is needed. If defect/ has images they become the test set, otherwise Anomalib
makes synthetic anomalies to report metrics. Trains on the Intel GPU (XPU) when available.
Output: models/anomaly/<model>/ with weights/openvino/model.xml, meta.json and the ROI.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from anomalib.data import Folder
from anomalib.deploy import CompressionType, ExportType
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Padim, Patchcore

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"padim": Padim, "patchcore": Patchcore}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="padim")
    ap.add_argument("--data", default=str(ROOT / "data/anomaly"))
    ap.add_argument("--out", default=str(ROOT / "models/anomaly"))
    ap.add_argument("--size", type=int, default=256, help="square input size baked into the export")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--int8", action="store_true", help="also export an INT8 (PTQ) model")
    ap.add_argument("--cpu", action="store_true", help="train on CPU even if XPU is available")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out) / args.model
    n_good = len(list((data / "good").glob("*.png")))
    n_defect = len(list((data / "defect").glob("*.png"))) if (data / "defect").exists() else 0
    if n_good < 10:
        raise SystemExit(f"only {n_good} good images in {data/'good'}, capture more first")
    print(f"data: {n_good} good, {n_defect} defect")

    dm = Folder(
        name="bricks", root=data, normal_dir="good",
        abnormal_dir="defect" if n_defect else None,
        test_split_mode="from_dir" if n_defect else "synthetic",
        train_batch_size=args.batch, eval_batch_size=args.batch, num_workers=4, seed=0,
    )
    use_xpu = torch.xpu.is_available() and not args.cpu
    device_kw = {"accelerator": XPUAccelerator(), "strategy": SingleXPUStrategy()} if use_xpu else {"accelerator": "cpu"}
    engine = Engine(max_epochs=1, default_root_dir=out / "runs", logger=False, **device_kw)
    model = MODELS[args.model]()

    t0 = time.time()
    engine.fit(model=model, datamodule=dm)
    train_s = time.time() - t0
    metrics = engine.test(model=model, datamodule=dm)[0]

    if out.joinpath("weights").exists():
        shutil.rmtree(out / "weights")
    size = (args.size, args.size)
    xml = engine.export(model, ExportType.OPENVINO, export_root=out, input_size=size)
    exports = {"fp32": str(xml)}
    if args.int8:
        int8_root = out / "int8"
        exports["int8"] = str(engine.export(model, ExportType.OPENVINO, export_root=int8_root, input_size=size,
                                            compression_type=CompressionType.INT8_PTQ, datamodule=dm))

    roi_file = data / "roi.json"
    meta = {
        "model": args.model, "input_size": size, "train_device": "xpu" if use_xpu else "cpu",
        "train_seconds": round(train_s, 1), "n_good": n_good, "n_defect": n_defect,
        "metrics": {k: float(v) for k, v in metrics.items()}, "exports": exports,
        "roi": json.loads(roi_file.read_text()) if roi_file.exists() else None,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
