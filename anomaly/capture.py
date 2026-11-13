"""Capture inspection-camera images for Anomalib.

Usage (on the Intel laptop, inside ~/checkurai):
    python anomaly/capture.py good   --count 60 --every 1.0
    python anomaly/capture.py defect --count 20 --every 2.0
    python anomaly/capture.py good   --roi 160,80,320,320     # crop to where the brick is shown

Images go to data/anomaly/<label>/ (gitignored). The ROI is saved to data/anomaly/roi.json
and train.py copies it next to the model, so detection crops exactly the same region.
Close the camdash browser tab first, otherwise the camera may be busy.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def find_cam(role: str) -> str:
    out = subprocess.run([sys.executable, str(ROOT / "tools/find_cams.py"), "--json"],
                         capture_output=True, text=True, check=True).stdout
    dev = json.loads(out).get(role)
    if not dev:
        sys.exit(f"camera role '{role}' not found (run tools/find_cams.py)")
    return dev


def open_cam(dev: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit(f"cannot open {dev} (busy? close camdash / Studio recording)")
    for _ in range(10):  # let auto exposure settle
        cap.read()
    return cap


def crop(frame, roi):
    if not roi:
        return frame
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", choices=["good", "defect"])
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--every", type=float, default=1.0, help="seconds between shots")
    ap.add_argument("--cam", default="inspection", help="camera role from tools/find_cams.py")
    ap.add_argument("--device", help="explicit /dev/videoN, overrides --cam")
    ap.add_argument("--roi", help="x,y,w,h crop in the 640x480 frame")
    ap.add_argument("--data", default=str(ROOT / "data/anomaly"))
    args = ap.parse_args()

    data = Path(args.data)
    out = data / args.label
    out.mkdir(parents=True, exist_ok=True)
    roi = [int(v) for v in args.roi.split(",")] if args.roi else None
    roi_file = data / "roi.json"
    if roi:
        roi_file.write_text(json.dumps(roi))
    elif roi_file.exists():
        roi = json.loads(roi_file.read_text())
        print(f"using saved ROI {roi} from {roi_file}")

    cap = open_cam(args.device or find_cam(args.cam))
    print(f"capturing {args.count} '{args.label}' images every {args.every}s -> {out}")
    try:
        for i in range(args.count):
            time.sleep(args.every)
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed, retrying")
                continue
            path = out / f"{time.strftime('%H%M%S')}_{i:03d}.png"
            cv2.imwrite(str(path), crop(frame, roi))
            print(f"[{i + 1}/{args.count}] {path.name}  (move/rotate the brick now)")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
