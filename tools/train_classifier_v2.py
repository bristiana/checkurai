"""Supervised good/defect classifier on the background-neutral per-view crops (data/inspection_crops_views_b2all):
ImageNet resnet18 fine-tuned on real clean crops (label 0) vs the same crops with SYNTHETIC marker strokes (label 1),
heavy photometric aug + synthetic specular highlights on goods. Real defects are never trained on (validation only)."""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "4"
import json, time
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torchvision

torch.set_num_threads(4)
R = Path("/home/intel/checkurai")
C = R / "data/inspection_crops_views_b2all"
NAME = os.environ.get("NAME", "resnet18_synth_v1")
EPOCHS = int(os.environ.get("EPOCHS", 10))
S = 224
dev = torch.device("xpu" if torch.xpu.is_available() and os.environ.get("DEV", "xpu") == "xpu" else "cpu")
rng = np.random.default_rng(int(os.environ.get("SEED", 0)))
torch.manual_seed(int(os.environ.get("SEED", 0)))
VIEWS = ("direct", "left", "right")
V2 = os.environ.get("SYNTH", "v1") == "v2"  # v2: blue-marker ink incl. saturated blue, dots from specks to big blobs


def load(p):
    return cv2.resize(cv2.imread(str(p)), (S, S), interpolation=cv2.INTER_AREA)


def brick_mask(img):
    m = (np.abs(img.astype(np.int16) - 128).max(2) > 14).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones(img.shape[:2], bool)
    c = max(cnts, key=cv2.contourArea)
    f = np.zeros_like(m)
    cv2.drawContours(f, [c], -1, 1, cv2.FILLED)
    return f > 0


def bezier(p, n=40):
    out = []
    for tt in np.linspace(0, 1, n):
        q = np.array(p, np.float32)
        while len(q) > 1:
            q = (1 - tt) * q[:-1] + tt * q[1:]
        out.append(q[0])
    return np.array(out, np.int32)


def synth_defect(img, mask):
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ys, xs = np.nonzero(inner if inner.sum() > 50 else mask)
    if len(xs) == 0:
        return img
    layer = np.zeros(img.shape[:2], np.float32)
    colmap = np.zeros(img.shape, np.float32)
    for _ in range(rng.integers(1, 4)):
        if V2 and rng.random() < 0.35:  # blue marker: dark or saturated royal blue (BGR)
            col = (np.array([rng.uniform(70, 120), rng.uniform(10, 40), rng.uniform(0, 30)]) if rng.random() < 0.4
                   else np.array([rng.uniform(130, 210), rng.uniform(30, 80), rng.uniform(0, 40)]))
        elif not V2 and rng.random() < 0.2:
            col = np.array([rng.uniform(70, 120), rng.uniform(10, 40), rng.uniform(0, 30)])  # dark blue ink (BGR)
        else:
            col = np.full(3, rng.uniform(0, 70))  # black / dark grey marker
        L = np.zeros(img.shape[:2], np.float32)
        i = rng.integers(len(xs))
        x0, y0 = int(xs[i]), int(ys[i])
        kind = rng.choice(["scribble", "dot", "line"], p=[0.4, 0.4, 0.2] if V2 else [0.5, 0.3, 0.2])
        if kind == "dot" and V2:  # tiny specks to big irregular blobs
            r = int(rng.integers(1, 15))
            cv2.ellipse(L, (x0, y0), (r, max(1, int(r * rng.uniform(0.55, 1)))), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        elif kind == "dot":
            cv2.circle(L, (x0, y0), int(rng.integers(2, 8)), 1.0, -1)
        elif kind == "line":
            a, ln = rng.uniform(0, np.pi), rng.uniform(8, 40)
            cv2.line(L, (x0, y0), (int(x0 + ln * np.cos(a)), int(y0 + ln * np.sin(a))), 1.0, int(rng.integers(1, 5)))
        else:
            pts = [(x0, y0)]
            for _ in range(rng.integers(2, 6)):
                pts.append((pts[-1][0] + int(rng.integers(-25, 26)), pts[-1][1] + int(rng.integers(-25, 26))))
            cv2.polylines(L, [bezier(pts)], False, 1.0, int(rng.integers(1, 5)))
        L = cv2.GaussianBlur(L, (3, 3), 0) * rng.uniform(0.6, 1.0) * mask
        upd = L > layer
        layer = np.maximum(layer, L)
        colmap[upd] = col
    out = img.astype(np.float32) * (1 - layer[..., None]) + colmap * layer[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def highlight(img, mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return img
    H = np.zeros(img.shape[:2], np.float32)
    for _ in range(rng.integers(1, 4)):
        i = rng.integers(len(xs))
        c = (int(xs[i]), int(ys[i]))
        if rng.random() < 0.6:
            cv2.ellipse(H, c, (int(rng.integers(3, 18)), int(rng.integers(2, 10))), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        else:
            a, ln = rng.uniform(0, np.pi), rng.uniform(10, 50)
            cv2.line(H, c, (int(c[0] + ln * np.cos(a)), int(c[1] + ln * np.sin(a))), 1.0, int(rng.integers(2, 7)))
    H = cv2.GaussianBlur(H, (0, 0), rng.uniform(1.5, 5)) * rng.uniform(0.3, 0.95) * mask
    out = img.astype(np.float32) * (1 - H[..., None]) + 255 * H[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def photometric(img, mask):
    f = img.astype(np.float32) / 255 * rng.uniform(0.6, 1.4)
    m = f[mask].mean() if mask.any() else 0.5
    f = (f - m) * rng.uniform(0.7, 1.3) + m
    f = np.clip(f, 0, 1) ** rng.uniform(0.7, 1.4)
    f = np.clip(f * rng.uniform(0.85, 1.15, 3) * 255, 0, 255)
    if rng.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        f = f + rng.normal(0, rng.uniform(2, 8), f.shape)
    f = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # imperfect outline: a band of paper-like texture around the brick
        band = cv2.dilate(mask.astype(np.uint8), np.ones((int(rng.integers(3, 12)),) * 2, np.uint8)) > 0
        paper = np.clip(rng.normal(rng.uniform(140, 200), 12, f.shape) * rng.uniform(0.9, 1.1, 3), 0, 255).astype(np.uint8)
        f[band & ~mask] = paper[band & ~mask]
        mask = band
    f[~mask] = 128
    return f


def geo(img):
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1]
    return np.ascontiguousarray(np.rot90(img, int(rng.integers(4))))


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_t(imgs):
    x = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]).astype(np.float32) / 255
    return torch.from_numpy(((x - MEAN) / STD).transpose(0, 3, 1, 2))


def sample(img, mask, defect):
    x = synth_defect(img, mask) if defect else img
    if rng.random() < (0.3 if defect else 0.5):
        x = highlight(x, mask)
    return geo(photometric(x, mask))


def val_set():
    rows = []
    for part in ("calib_good", "test_good", "defect"):
        by = {}
        for p in sorted((C / part).glob("*.jpg")):
            stem, v = p.stem.rsplit("__", 1)
            by.setdefault(stem, {})[v] = load(p)
        for stem, d in by.items():
            rows.append((part, stem, [d[v] for v in VIEWS]))
    return rows


def build():
    net = torchvision.models.resnet18()
    net.fc = nn.Linear(512, 1)
    return net


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    train = [load(p) for p in sorted((C / "train_good").glob("*.jpg"))]
    val = val_set()
    import sys as _sys
    _sys.path.insert(0, str(R))
    from tools.brick_classifier import BrickClassifierInspector
    _ins = BrickClassifierInspector("resnet18_synth_v1", "CPU")
    LIVE_D = ["20261114-203554", "20261114-214717", "20261114-215341", "20261114-215610", "20261114-215746",
              "20261114-220809"]
    LIME = ["20261114-221034", "20261114-221155", "20261114-221313"]
    LIVE_C = ["20261114-205215", "20261114-211840", "20261114-214312"]
    HOLD = os.environ.get("HOLD_LIME", "0") == "1"

    def live_views(run):
        f = cv2.imread(str(R / "data/cell_runs" / run / "cycle_000_frame.jpg"))
        w = _ins.cropper.windows(f)
        return [cv2.resize(w[v][0], (S, S), interpolation=cv2.INTER_AREA) for v in VIEWS]

    for run in (LIVE_C if os.environ.get("BAGS", "1") == "1" else []):  # live clean bricks: extra goods
        train += live_views(run) * 3
    tmask = [brick_mask(i) for i in train]
    bags = [r[2] for r in val if r[0] == "defect"] + [live_views(r) for r in LIVE_D + ([] if HOLD else LIME)]
    bmask = [[brick_mask(v) for v in b] for b in bags]
    lime_rows = [live_views(r) for r in LIME]
    clean_rows = [live_views(r) for r in LIVE_C]
    print(f"real defect bags {len(bags)} (hold lime {HOLD}), live clean views {3 * len(LIVE_C)}", flush=True)
    good = np.array([r[0] != "defect" for r in val])
    print(f"train goods {len(train)} crops | val photos {len(val)} ({(~good).sum()} real defects) | device {dev}", flush=True)
    net = torchvision.models.resnet18()
    net.load_state_dict(torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")))
    net.fc = nn.Linear(512, 1)
    net = net.to(dev)
    opt = torch.optim.AdamW([{"params": [p for n, p in net.named_parameters() if not n.startswith("fc")], "lr": 2e-4},
                             {"params": net.fc.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    lossf = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def predict(imgs):
        net.eval()
        out = []
        for i in range(0, len(imgs), 48):
            out.append(torch.sigmoid(net(to_t(imgs[i:i + 48]).to(dev))).float().cpu().numpy()[:, 0])
        return np.concatenate(out)

    def evaluate():
        p = predict([v for r in val for v in r[2]]).reshape(len(val), 3)
        pm = p.max(1)
        auc = roc_auc_score(~good, pm)
        thr = float(np.sort(pm[good])[-2]) + 1e-6  # <= 1 held-out good false alarm
        return auc, thr, pm, p

    for ep in range(EPOCHS):
        net.train()
        t0 = time.time()
        idx = rng.permutation(len(train))
        tl = []
        for b in range(0, len(idx), 16):
            ii = idx[b:b + 16]
            xs = [sample(train[i], tmask[i], False) for i in ii] + [sample(train[i], tmask[i], True) for i in ii]
            y = torch.tensor([0.0] * len(ii) + [1.0] * len(ii), device=dev)[:, None]
            loss = lossf(net(to_t(xs).to(dev)), y)
            if os.environ.get("BAGS", "1") == "1":
                bi = rng.choice(len(bags), 6, replace=False)  # real defects: max over the 3 views must say defect
                bx = [geo(photometric(v, m)) for k in bi for v, m in zip(bags[k], bmask[k])]
                bl = net(to_t(bx).to(dev)).view(len(bi), 3).max(1).values[:, None]
                loss = loss + lossf(bl, torch.ones_like(bl))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(float(loss))
        sched.step()
        auc, thr, pm, _ = evaluate()
        pl = predict([v for r in lime_rows for v in r]).reshape(len(lime_rows), 3).max(1)
        pc = predict([v for r in clean_rows for v in r]).reshape(len(clean_rows), 3).max(1)
        print(f"   lime-dot live {np.round(pl, 3).tolist()} | live clean {np.round(pc, 3).tolist()}", flush=True)
        print(f"epoch {ep} loss {np.mean(tl):.3f} val AUROC {auc:.3f} | thr(<=1 good FA) {thr:.3f}: real defects "
              f"{int((pm[~good] >= thr).sum())}/{(~good).sum()}, goods FA {int((pm[good] >= thr).sum())}/{good.sum()} | "
              f"@0.5: defects {int((pm[~good] >= 0.5).sum())} FA {int((pm[good] >= 0.5).sum())} | {time.time() - t0:.0f}s", flush=True)

    out = R / "models/classifier" / NAME
    out.mkdir(parents=True, exist_ok=True)
    auc, thr, pm, p = evaluate()
    net.cpu()
    torch.save(net.state_dict(), out / "model.pt")
    sp = {r["file"].rsplit(".", 1)[0]: r for r in json.load(open(R / "results/anomaly/split_views_b2all.json"))}
    types = {}
    for (part, stem, _), s in zip(val, pm):
        if part == "defect":
            types.setdefault(sp[stem].get("type"), []).append(float(s))
    meta = {"arch": "resnet18", "size": S, "synth": "v2" if V2 else "v1", "seed": int(os.environ.get("SEED", 0)), "threshold": thr, "threshold_rule": "<= 1 held-out good false alarm (validation)",
            "views": list(VIEWS), "fusion": "max over views of P(defect)", "crop_set": "inspection_crops_views_b2all",
            "trained_on": "train_good crops of b2all + 3 live clean frames vs synthetic v2 marker strokes/specks, plus REAL defect bags (24 photo defects + live defect cycles; loss on max over views); val defects are therefore TRAINED ON", "hold_lime": HOLD,
            "val": {"auroc": float(auc), "goods": int(good.sum()), "defects": int((~good).sum()),
                    "goods_fa_at_thr": int((pm[good] >= thr).sum()),
                    "recall_at_thr": {t: f"{sum(x >= thr for x in v)}/{len(v)}" for t, v in types.items()},
                    "scores": [(r[0], r[1], round(float(s), 4)) for r, s in zip(val, pm)]}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("saved", out, "thr", round(thr, 4), meta["val"]["recall_at_thr"], flush=True)
