"""Average of the seed classifiers' P(defect) as one model (models/classifier/NAME, output = probability):
validation on the held-out b2all goods + real defects, same threshold rule (<= 1 good FA), OpenVINO IR."""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "4"
import sys, json, shutil
import numpy as np, torch
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/intel/checkurai")
import cls_train as ct  # noqa: E402
from tools.brick_classifier import load_torch, MODELS  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

torch.set_num_threads(4)
NAME = sys.argv[1]
MEMBERS = sys.argv[2:]


class Ens(torch.nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)

    def forward(self, x):
        return torch.stack([torch.sigmoid(n(x)) for n in self.nets]).mean(0)


net = Ens([load_torch(MODELS / m) for m in MEMBERS]).eval()
val = ct.val_set()
good = np.array([r[0] != "defect" for r in val])
with torch.no_grad():
    imgs = [v for r in val for v in r[2]]
    p = np.concatenate([net(ct.to_t(imgs[i:i + 48])).numpy()[:, 0] for i in range(0, len(imgs), 48)]).reshape(len(val), 3)
pm = p.max(1)
auc = roc_auc_score(~good, pm)
thr = float(np.sort(pm[good])[-2]) + 1e-6
out = MODELS / NAME
out.mkdir(parents=True, exist_ok=True)
for f in (MODELS / MEMBERS[0]).iterdir():
    if f.name == "crop.json" or "_empty" in f.name:
        shutil.copy(f, out / f.name)
import openvino as ov  # noqa: E402
ov.save_model(ov.convert_model(net, example_input=torch.zeros(1, 3, ct.S, ct.S), input=[1, 3, ct.S, ct.S]), out / "model.xml")
sp = {r["file"].rsplit(".", 1)[0]: r for r in json.load(open(ct.R / "results/anomaly/split_views_b2all.json"))}
types = {}
for (part, stem, _), s in zip(val, pm):
    if part == "defect":
        types.setdefault(sp[stem].get("type"), []).append(float(s))
meta = {"arch": f"ensemble of {len(MEMBERS)} resnet18 (mean P(defect))", "members": MEMBERS, "size": ct.S, "output": "probability",
        "threshold": thr, "threshold_rule": "<= 1 held-out good false alarm (validation)", "views": list(ct.VIEWS),
        "fusion": "max over views", "crop_set": "inspection_crops_views_b2all",
        "val": {"auroc": float(auc), "goods": int(good.sum()), "defects": int((~good).sum()),
                "goods_fa_at_thr": int((pm[good] >= thr).sum()),
                "recall_at_thr": {t: f"{sum(x >= thr for x in v)}/{len(v)}" for t, v in types.items()},
                "scores": [(r[0], r[1], round(float(s), 4)) for r, s in zip(val, pm)]}}
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"{NAME}: val AUROC {auc:.3f} thr {thr:.4f} real defects {int((pm[~good] >= thr).sum())}/{(~good).sum()} "
      f"{meta['val']['recall_at_thr']} goods FA {meta['val']['goods_fa_at_thr']}/{good.sum()} | @0.5 defects "
      f"{int((pm[~good] >= 0.5).sum())} FA {int((pm[good] >= 0.5).sum())}", flush=True)
