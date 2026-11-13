#!/usr/bin/env python3
"""Make a printable A4 sheet of ArUco markers for the table (camera-to-robot calibration and camera-moved check).

    python tools/make_markers.py                 # -> markers/aruco_a4.pdf (+ .png)
    python tools/make_markers.py --size-mm 50 --ids 0 1 2 3

Print at 100 % scale ("actual size", no "fit to page"), then check the 100 mm scale bar with a ruler.
Dictionary: DICT_4X4_50. Each marker has a white quiet zone; cut along the dashed lines, keep the white border.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DPI = 300
A4_MM = (210, 297)


def mm(v: float) -> int:
    return int(round(v / 25.4 * DPI))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size-mm", type=float, default=50.0, help="black square side length in mm")
    ap.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "markers" / "aruco_a4")
    args = ap.parse_args()

    aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    page = np.full((mm(A4_MM[1]), mm(A4_MM[0])), 255, np.uint8)
    side, quiet, gap = mm(args.size_mm), mm(8), mm(14)
    cell = side + 2 * quiet
    x0, y0 = mm(15), mm(25)
    for k, mid in enumerate(args.ids):
        r, c = divmod(k, 2)
        x, y = x0 + c * (cell + gap), y0 + r * (cell + gap + mm(6))
        if y + cell > page.shape[0] - mm(30):
            raise SystemExit("too many markers for one page; pass fewer --ids or a smaller --size-mm")
        page[y + quiet:y + quiet + side, x + quiet:x + quiet + side] = cv2.aruco.generateImageMarker(aruco, mid, side)
        for t in range(0, cell, mm(4)):  # dashed cut lines
            page[y, x + t:x + t + mm(2)] = 0
            page[y + cell - 1, x + t:x + t + mm(2)] = 0
            page[y + t:y + t + mm(2), x] = 0
            page[y + t:y + t + mm(2), x + cell - 1] = 0
        cv2.putText(page, f"id {mid}  ({args.size_mm:g} mm)", (x, y + cell + mm(5)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    by = page.shape[0] - mm(18)
    page[by:by + mm(1.5), x0:x0 + mm(100)] = 0
    cv2.putText(page, "100 mm - check with a ruler. DICT_4X4_50", (x0, by - mm(3)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    png = args.out.with_suffix(".png")
    cv2.imwrite(str(png), page)
    try:
        from PIL import Image
        Image.open(png).save(args.out.with_suffix(".pdf"), resolution=DPI)
        print(f"wrote {args.out.with_suffix('.pdf')} and {png}")
    except ImportError:
        print(f"wrote {png} (install pillow for the PDF); print it at {DPI} dpi")


if __name__ == "__main__":
    main()
