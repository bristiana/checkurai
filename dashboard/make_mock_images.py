"""Build mock images for the dashboard from real station photos in mock/raw/ (dev only)."""
import glob
import os

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "mock", "raw")
OUT = os.path.join(HERE, "mock", "img")
os.makedirs(OUT, exist_ok=True)

VIEWS = {"direct": (732, 508, 1460, 1080), "left": (272, 72, 848, 492), "right": (1026, 40, 1578, 476)}
# anchor colours of a warm "heat" ramp: transparent -> deep red -> orange -> pale yellow
RAMP = np.array([[0, 0, 0], [120, 20, 30], [220, 70, 20], [250, 170, 40], [255, 245, 200]], float)


def heat_rgb(m):
    x = np.clip(m, 0, 1) * (len(RAMP) - 1)
    i = np.clip(x.astype(int), 0, len(RAMP) - 2)
    f = (x - i)[..., None]
    return RAMP[i] * (1 - f) + RAMP[i + 1] * f


def heat_overlay(img, blobs, seed):
    rng = np.random.default_rng(seed)
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.zeros((h, w))
    m += 0.12 * rng.random((h, w))  # faint texture
    for cx, cy, r, a in blobs:
        m += a * np.exp(-(((xx - cx * w) ** 2 + (yy - cy * h) ** 2) / (2 * (r * w) ** 2)))
    m = np.array(Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))) / 255
    base = np.asarray(img.convert("RGB"), float)
    alpha = np.clip((m - 0.3) / 0.45, 0, 1)[..., None] * 0.8  # transparent where normal, like the real overlay
    out = base * (1 - alpha) + heat_rgb(m) * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def save(im, name, size=None):
    if size:
        im = im.copy()
        im.thumbnail(size)
    im.convert("RGB").save(os.path.join(OUT, name), quality=86)


# top + wrist camera frames (two versions so the mock can alternate)
tw = Image.open(os.path.join(RAW, "top_warp_act-pick-v3.jpg"))
save(tw.crop((640, 40, 1280, 480)), "top_a.jpg")  # skip the burnt-in panel caption
save(tw.crop((0, 40, 640, 480)), "top_b.jpg")
save(tw.crop((300, 180, 520, 345)).resize((640, 480)), "wrist_a.jpg")
save(tw.crop((960, 200, 1180, 365)).resize((640, 480)), "wrist_b.jpg")

# inspection views: one set per photo, defect photos get a hot spot on the worst view
photos = sorted(glob.glob(os.path.join(RAW, "good_*.jpg")))[:3] + sorted(glob.glob(os.path.join(RAW, "defect_*.jpg")))[:3]
def square(im, box):
    x0, y0, x1, y1 = box
    s = min(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    return im.crop((cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2)).resize((384, 384))


# hot spots per (set, view): (x, y, sigma, strength), normalised; the mock server rings the same spots (r = 2 sigma)
HOT = {
    (3, "left"): [(0.42, 0.50, 0.055, 1.0), (0.62, 0.30, 0.03, 0.55)],
    (3, "direct"): [(0.40, 0.45, 0.045, 0.7)],
    (4, "direct"): [(0.35, 0.365, 0.045, 1.0), (0.57, 0.515, 0.04, 0.8)],
    (4, "left"): [(0.60, 0.45, 0.04, 0.65)],
    (2, "right"): [(0.45, 0.55, 0.05, 0.75)],
}
REAL = os.path.join(OUT, "real")
real = {v: Image.open(os.path.join(REAL, f"view_{v}.jpg")) for v in VIEWS} if os.path.isdir(REAL) else {}
for k, p in enumerate(photos):
    im = Image.open(p)
    defect = "defect_" in os.path.basename(p)
    for v, box in VIEWS.items():
        crop = real[v] if (real and k in (0, 3)) else square(im, box)
        save(crop, f"set{k}_view_{v}.jpg")
        blobs = [(0.5, 0.45, 0.10, 0.18)] + HOT.get((k, v), [])
        save(heat_overlay(crop, blobs, k * 10 + len(v)), f"set{k}_heat_{v}.jpg")
    th = real["direct"] if (real and k in (0, 3)) else square(im, VIEWS["direct"])
    save(th.resize((160, 160)), f"set{k}_thumb.jpg")
print(sorted(os.listdir(OUT)))
