"""Bake fixes into a trained policy's OpenVINO export: a wider gripper opening, and a top-camera correction.

Gripper: in the demos the jaws open to only ~26 (place) / ~30 (pick) around a brick ~15 wide. The policy's
grasp lands a little off to one side, so a jaw hits the brick and pushes it away. A wider opening gives it
room. Commands below --grip-from (closing on and squeezing the brick) are left alone, so the grip force is
unchanged:

    g' = min(g + (gain - 1) * max(g - from, 0), max)      (Studio units, 0 closed .. 100 open)

Top camera (--top-warp auto): the top camera drifts (it tilted ~40 px over the evening, also while demos were
being recorded). So every model saw the table from its own camera position. For each model, the ArUco markers
are measured in the top-camera video of the exact demos it was trained on (Studio's training snapshot; median
over the episodes = the centre of what the model saw). They are then compared with the markers in the live top
camera. The live picture is warped onto that model's training view: a GridSample op in front of its
images.top input. The policy then sees the table, bowls and spot where it learned them, even if the camera
moved a little since. Re-run after any bump.

Plate follow (--plate-follow, off by default, pick models): the pick policy barely follows a moved plate. Shifting
the plate (with its brick) 20 px in its top picture moves the predicted grasp by only ~5-15 % of what the demos
show for such a move, so a plate 2 cm off its demo spot means a 2 cm miss. This measures the plate in the live
top camera now (in the model's training view, vs its demo median) and adds the missing part of the grasp pose
to the actions: d px -> gain * slope * d on pan / lift / elbow (slopes fitted on act-pick-v4b's demos, grasp
pose vs brick position). It is gated to approach + grasp: on only for chunk steps where the gripper has been
opened (state, or earlier in the chunk) and the arm is on the plate side (state pan and that step's pan above
the carry towards the spot), so the carry, the spot and the return HOME are unchanged. Not stored: give the
flag again on every tune that should keep it.

Paint plate (--paint-plate, off by default, place models): the yellow plate stood empty on its spot in every place
demo and is gone from the cell now; the place policies hesitate without it. This paints the model's own demo plate
(top_ref.png, mapped onto its training view) back into its top picture, after the warp: only inside the plate's
demo disc (top_ref.json bowls.yellow, + PP_GROW px) and only where the live pixel is bare table (low colour,
table brightness); the arm, a brick, a hand are kept, a shadow darkens the painted plate; off while the real
plate is in the disc. Not stored either.

Hide bowl bricks (--hide-bowl-bricks, off by default, place models): the place demos had empty bowls; the sorted
bricks pile up in them and the place policy hesitates over a full bowl. This finds the bricks lying in the lilac and
orange bowls in one live top picture and shows the policy that picture with them inpainted, but only while those
pixels still look as they did (the arm or a new brick over them stays visible). Made right before each place stage
(tools/orchestrator.py --hide-bowl-bricks). Not stored either. Once an empty-bowl photo exists
(--capture-empty-bowls -> config/cell_ref/empty_bowls_top.png), it pastes the whole bowls from that photo instead
of inpainting the bricks it finds (--hide-mode inpaint: the old way).

Studio runs the exported graph as it is, so the fixes apply to every run of that model (the cell program and
Studio's inference page). The trained export is kept as <graph>.orig.xml/.bin (act, smolvla). Every tune rebuilds from that
backup, so tunes never stack. Run with Studio's venv (same OpenVINO as Studio):

    PY=~/physical-ai-studio/application/backend/.venv/bin/python
    $PY tools/policy_tune.py act-place-good-v3                     # gripper 1.5 (or the stored values) + stored warp
    $PY tools/policy_tune.py act-place-good-v3 --top-warp auto     # + measure the top camera now and undo its move
    $PY tools/policy_tune.py --cell --top-warp auto                # all models in config/cell.json (after a bump)
    $PY tools/policy_tune.py --camera-report [model ...]           # how far the camera is from each model's demos
    $PY tools/policy_tune.py act-pick-v4b --top-warp auto --plate-follow   # + follow a moved plate (pick)
    $PY tools/policy_tune.py act-place-good-v5 --paint-plate       # + paint the demo plate back in (place)
    $PY tools/policy_tune.py act-place-good-v5 --hide-bowl-bricks  # + hide the bricks lying in the bowls (place)
    $PY tools/policy_tune.py --capture-empty-bowls                 # photo of the empty bowls for that (once)
    $PY tools/policy_tune.py act-place-good-v3 --show              # current tune
    $PY tools/policy_tune.py act-place-good-v3 --restore           # back to the trained export
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import openvino.opset9 as ops9

ROOT = Path(__file__).resolve().parent.parent
GRIPPER = 5  # action order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
GRIP_DEFAULT = {"grip_gain": 1.5, "grip_from": 12.0, "grip_max": 60.0}
STUDIO = Path.home() / ".local/share/physicalai"
TOP = "observation.images.top"
CAM_W, CAM_H, NET = 640, 480, 384  # top-camera frame; letterboxed network input (384x288 + 48 px bars)
CELL_KEYS = ("pick_model", "place_good_model", "place_bad_model")


@dataclass
class Geo:
    """How an export turns a 640x480 top-camera frame into its network input, read from its manifest.json."""
    policy: str
    xml: str  # the OpenVINO graph file name (act.xml, smolvla.xml)
    net_w: int
    net_h: int
    w_net: int  # the resized picture inside the network input, before padding
    h_net: int
    pad_top: int
    pad_left: int
    lo: float  # pixel value of black: 0 for [0, 1] inputs (ACT), -1 for [-1, 1] (SmolVLA)
    slot: int | None  # top camera's slot in a stacked "images" input (SmolVLA); None = its own images.top input
    act_norm: tuple[np.ndarray, np.ndarray] | None  # action (mean, std) if the graph outputs normalised actions


def export_geo(d: Path) -> Geo:
    man = json.loads((d / "manifest.json").read_text())
    model, pol = man["model"], man["policy"]["name"]
    pre = {p["type"]: p for p in model.get("preprocessors", [])}
    cams = [f["init_args"]["name"] for f in model["input_features"] if f["init_args"]["ftype"] == "VISUAL"]
    norm = None
    for p in model.get("postprocessors", []):
        if p["type"] == "denormalize" and p.get("mode") == "mean_std" and "action" in p.get("stats", {}):
            st = p["stats"]["action"]
            norm = (np.array(st["mean"], np.float32), np.array(st["std"], np.float32))
        elif p["type"] == "denormalize":
            raise SystemExit(f"unsupported action denormalisation {p.get('mode')!r} in {d}")
    xml = model["artifacts"]["openvino"]
    if "smolvla_resize" in pre:  # resize keeping aspect, pad top + left, then [0,1] -> [-1,1]
        h, w = pre["smolvla_resize"]["image_resolution"]
        r = max(CAM_W / w, CAM_H / h)
        wn, hn = max(1, min(int(CAM_W / r), w)), max(1, min(int(CAM_H / r), h))
        order = pre["smolvla_resize"].get("image_key_reorder_map") or {}
        slot = order.get("images.top", order.get("top")) if order else cams.index("images.top")
        return Geo(pol, xml, w, h, wn, hn, h - hn, w - wn, -1.0, slot, norm)
    if pre.get("resize", {}).get("mode") == "letterbox":  # resize keeping aspect, pad centred, [0,1]
        h, w = pre["resize"]["image_resolution"]
        r = max(CAM_W / w, CAM_H / h)
        wn, hn = round(CAM_W / r), round(CAM_H / r)
        return Geo(pol, xml, w, h, wn, hn, (h - hn) // 2, (w - wn) // 2, 0.0, None, norm)
    raise SystemExit(f"unsupported export in {d}: policy {pol}, preprocessors {list(pre)}")


def cfg() -> dict:
    return json.loads((ROOT / "config" / "cell.json").read_text())


def model_record(name: str) -> dict:
    c = cfg()
    url = f"{c.get('url', 'http://127.0.0.1:7860')}/api/projects/{c['project']}/models"
    items = json.load(urllib.request.urlopen(url, timeout=30))
    items = items.get("items", items) if isinstance(items, dict) else items
    hit = [m for m in items if name in (m.get("name"), m.get("id"))]
    if not hit:
        raise SystemExit(f"no model {name!r}; have: {', '.join(m.get('name', '?') for m in items)}")
    return hit[0]


def export_dir(rec: dict) -> Path:
    return Path(rec["path"]) / "exports" / "openvino"


# ---- gripper ----------------------------------------------------------------------------------------------
def grip_map(src: ov.Output, gain: float, grip_from: float, grip_max: float) -> ov.Node:
    """The gripper map as graph ops on an action tensor [..., joints]; the other joints pass through."""
    n = src.get_partial_shape()[-1].get_length()
    vec = lambda fill, g: ops.constant(np.array([fill] * GRIPPER + [g] + [fill] * (n - GRIPPER - 1), np.float32))
    excess = ops.maximum(ops.subtract(src, vec(0.0, grip_from)), vec(0.0, 0.0))
    return ops.minimum(ops.add(src, ops.multiply(excess, vec(0.0, gain - 1.0))), vec(1e6, grip_max))


# ---- top camera -------------------------------------------------------------------------------------------
def detect(img_bgr: np.ndarray) -> dict[int, np.ndarray]:
    """ArUco marker id -> its 4 corners [4, 2] in camera pixels."""
    import cv2

    sys.path.insert(0, str(ROOT / "tools"))
    from cell_check import DET

    if img_bgr.shape[:2] != (CAM_H, CAM_W):
        raise SystemExit(f"top camera frame is {img_bgr.shape[1]}x{img_bgr.shape[0]}, expected {CAM_W}x{CAM_H}")
    corners, ids, _ = DET.detectMarkers(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    return {} if ids is None else {int(i): c[0].astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def median_markers(seen: list[dict[int, np.ndarray]], min_frac: float) -> dict[int, np.ndarray]:
    ids = {i for s in seen for i in s}
    out = {}
    for i in sorted(ids):
        cs = [s[i] for s in seen if i in s]
        if len(cs) >= max(1, min_frac * len(seen)):
            out[i] = np.median(np.stack(cs), 0)
    return out


BOWL_HSV = {"yellow": ((20, 120, 120), (35, 255, 255)), "lilac": ((115, 25, 90), (150, 255, 255)),
            "orange": ((5, 140, 120), (18, 255, 255))}  # OpenCV HSV ranges of the three bowls


def find_bowl(img: np.ndarray, color: str) -> tuple[np.ndarray, float] | None:
    """Centre + radius of the biggest blob of that colour (circle fit on its outline, ignoring the picture edge)."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo, hi = BOWL_HSV[color]
    m = cv2.inRange(hsv, np.array(lo), np.array(hi))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 4000:
        return None
    p = c[:, 0, :].astype(np.float64)
    h, w = img.shape[:2]
    p = p[(p[:, 0] > 3) & (p[:, 0] < w - 4) & (p[:, 1] > 3) & (p[:, 1] < h - 4)]
    if len(p) < 30:
        return None
    for _ in range(2):  # circle fit, then once more without the outline points that are far off (a brick, a hand)
        A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
        cx, cy, k = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
        r = float(np.sqrt(k + cx ** 2 + cy ** 2))
        d = np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r)
        p = p[d < max(4.0, np.percentile(d, 80))]
    return np.array([cx, cy]), r


def snapshot_dir(rec: dict) -> Path:
    hits = sorted((STUDIO / "snapshots").glob(f"*{rec['train_job_id']}"))
    if hits:
        return hits[0]
    d = STUDIO / "datasets" / rec["dataset_id"]  # no snapshot kept: the dataset now (may have more episodes)
    print(f"  (no training snapshot for {rec['name']}; using today's dataset {d.name[:8]})")
    return d


def video_frame(path: Path, t: float) -> np.ndarray | None:
    import cv2

    png = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR) if png else None


def train_top_ref(rec: dict) -> dict:
    """Markers in the top camera at the start of every training demo of this model (cached next to the export)."""
    cache = export_dir(rec) / "top_ref.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import pandas as pd

    snap = snapshot_dir(rec)
    eps = pd.concat([pd.read_parquet(f) for f in sorted((snap / "meta" / "episodes").rglob("*.parquet"))])
    eps = eps.sort_values("episode_index")
    import cv2

    per_ep, frames_ = [], []
    for _, e in eps.iterrows():
        v = snap / "videos" / TOP / f"chunk-{int(e[f'videos/{TOP}/chunk_index']):03d}" / f"file-{int(e[f'videos/{TOP}/file_index']):03d}.mp4"
        f = video_frame(v, float(e[f"videos/{TOP}/from_timestamp"]) + 0.3)  # arm at HOME, markers uncovered
        per_ep.append(detect(f) if f is not None else {})
        frames_.append(f)
    ref = median_markers(per_ep, 0.5)
    if len(ref) < 2:
        raise SystemExit(f"{rec['name']}: markers not found in its training video ({snap})")
    info = {}
    for i, c in ref.items():
        cen = np.array([s[i].mean(0) for s in per_ep if i in s])
        dev = np.linalg.norm(cen - c.mean(0), axis=1)
        info[str(i)] = {"corners": c.round(2).tolist(), "episodes": len(cen),
                        "spread_p90_px": round(float(np.percentile(dev, 90)), 1),
                        "first_ep": cen[0].round(1).tolist(), "last_ep": cen[-1].round(1).tolist()}
    # where each bowl stood, in this reference view (each demo's picture mapped by its own markers)
    bowls, cam_off = {c: [] for c in BOWL_HSV}, []
    for m, f in zip(per_ep, frames_):
        try:
            w = fit_warp(m, ref)
        except SystemExit:
            cam_off.append(1e9)
            continue
        cam_off.append(w["moved_px"])
        for c in BOWL_HSV:
            b = find_bowl(f, c)
            if b is not None:
                p = cv2.perspectiveTransform(b[0][None, None], np.array(w["H"]))[0, 0]
                bowls[c].append([p[0], p[1], b[1]])
    binfo = {}
    for c, v in bowls.items():
        if len(v) >= 0.5 * len(per_ep):
            v = np.array(v)
            med = np.median(v, 0)
            dev = np.linalg.norm(v[:, :2] - med[:2], axis=1)
            keep = v[dev <= np.percentile(dev, 95) + 3]  # drop a stray detection or two
            hull = cv2.convexHull(keep[:, :2].astype(np.float32))[:, 0]
            binfo[c] = {"c": med[:2].round(1).tolist(), "r": round(float(med[2]), 1), "episodes": len(v),
                        "spread_p75_px": round(float(np.percentile(dev, 75)), 1),
                        "hull": hull.round(1).tolist()}  # every bowl centre position seen in the demos
    best = int(np.argmin(cam_off))  # the demo picture closest to the reference view, for previews
    if frames_[best] is not None:
        cv2.imwrite(str(export_dir(rec) / "top_ref.png"), frames_[best])
    out = {"model": rec["name"], "source": snap.name, "episodes": len(per_ep), "markers": info, "bowls": binfo,
           "made_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    cache.write_text(json.dumps(out, indent=1) + "\n")
    return out


def ref_corners(ref: dict) -> dict[int, np.ndarray]:
    return {int(i): np.array(m["corners"]) for i, m in ref["markers"].items()}


def live_top(n: int = 12) -> dict[int, np.ndarray]:
    """Median marker corners over n frames (~1 s); a marker that moves between frames (a hand on it, a bowl
    being pushed against it) is left out."""
    sys.path.insert(0, str(ROOT / "tools"))
    import cam_lock as L

    if "top" not in L.SHARED:
        L.open_source("top")
    seen = [detect(f) for f in L.frames("top", n, gap=0.08)]
    live = median_markers(seen, 0.6)
    for i in list(live):
        cs = np.stack([s[i] for s in seen if i in s])
        wobble = float(np.abs(cs - live[i]).max())
        if wobble > 2.0:
            print(f"  (marker {i} moved {wobble:.0f} px while measuring - left out)")
            del live[i]
    if not live:
        raise SystemExit("top camera: no steady ArUco markers visible")
    return live


def fit_warp(live: dict[int, np.ndarray], ref: dict[int, np.ndarray]) -> dict:
    """Homography live -> training view from the marker corners (similarity if only 2-3 markers are shared)."""
    import cv2

    common = sorted(set(live) & set(ref))
    if len(common) < 2:
        raise SystemExit(f"top camera: only markers {common} match the training view (need 2+ of {sorted(ref)}); "
                         "point the camera roughly right first (tools/aim_view.py)")
    src = np.concatenate([live[i] for i in common])
    dst = np.concatenate([ref[i] for i in common])
    if len(common) >= 3:  # 12+ corners spread over the table: a full perspective fit is well determined
        H, _ = cv2.findHomography(src, dst, 0)
        kind = "perspective"
    else:
        A, _ = cv2.estimateAffinePartial2D(src, dst)
        H, kind = np.vstack([A, [0, 0, 1]]), "similarity"
    resid = float(np.linalg.norm(cv2.perspectiveTransform(src[None], H)[0] - dst, axis=1).max())
    moved = float(max(np.linalg.norm(live[i].mean(0) - ref[i].mean(0)) for i in common))
    # how much of the training view has no live picture behind it (comes out black)
    u, v = np.meshgrid(np.arange(0, CAM_W, 4.0), np.arange(0, CAM_H, 4.0))
    lp = cv2.perspectiveTransform(np.stack([u, v], -1).reshape(1, -1, 2), np.linalg.inv(H))[0]
    lost = float(((lp[:, 0] < 0) | (lp[:, 0] > CAM_W - 1) | (lp[:, 1] < 0) | (lp[:, 1] > CAM_H - 1)).mean())
    return {"H": H.tolist(), "kind": kind, "markers": common, "moved_px": round(moved, 1),
            "residual_px": round(resid, 2), "black_border_pct": round(100 * lost, 1)}


ACT_GEO = Geo("act", "act.xml", NET, NET, NET, round(CAM_H * NET / CAM_W), 0, 0, 0.0, None, None)
ACT_GEO.pad_top = (NET - ACT_GEO.h_net) // 2


def warp_grid(H: np.ndarray, g: Geo = ACT_GEO) -> np.ndarray:
    """GridSample grid [1, net_h, net_w, 2]: for each pixel of the network input in the training view, where to
    sample the network input made from the live picture (normalised, align_corners=False)."""
    sx, sy = CAM_W / g.w_net, CAM_H / g.h_net
    u, v = np.meshgrid(np.arange(g.net_w, dtype=np.float64), np.arange(g.net_h, dtype=np.float64))
    x = (u - g.pad_left + 0.5) * sx - 0.5
    y = (v - g.pad_top + 0.5) * sy - 0.5
    hi = np.linalg.inv(H)
    w = hi[2, 0] * x + hi[2, 1] * y + hi[2, 2]
    xl = (hi[0, 0] * x + hi[0, 1] * y + hi[0, 2]) / w
    yl = (hi[1, 0] * x + hi[1, 1] * y + hi[1, 2]) / w
    # outside the live camera picture -> black (as the padding bars and GridSample's zero padding)
    out = (xl < -0.5) | (xl > CAM_W - 0.5) | (yl < -0.5) | (yl > CAM_H - 0.5)
    gx = (2 * ((xl + 0.5) / sx - 0.5 + g.pad_left) + 1) / g.net_w - 1
    gy = (2 * ((yl + 0.5) / sy - 0.5 + g.pad_top) + 1) / g.net_h - 1
    bars = (v < g.pad_top) | (v >= g.pad_top + g.h_net) | (u < g.pad_left) | (u >= g.pad_left + g.w_net)
    gx[bars | out], gy[bars | out] = -3.0, -3.0
    return np.stack([gx, gy], -1)[None].astype(np.float32)


def grid_op(x: ov.Output, H: np.ndarray, g: Geo = ACT_GEO) -> ov.Node:
    """GridSample on a [1, 3, h, w] picture; regions with no live picture come out black (g.lo)."""
    lo = ops.constant(np.float32(g.lo))
    xin = ops.subtract(x, lo).output(0) if g.lo else x  # zero padding must mean black
    gs = ops9.grid_sample(xin, ops.constant(warp_grid(H, g)),
                          {"align_corners": False, "mode": "bilinear", "padding_mode": "zeros"})
    return ops.add(gs, lo) if g.lo else gs


def net_input(img_bgr: np.ndarray, g: Geo) -> np.ndarray:
    """The network input [1, 3, h, w] the export's preprocessing makes from a camera frame (RGB, [0,1] or [-1,1])."""
    import cv2

    a = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255, (g.w_net, g.h_net),
                   interpolation=cv2.INTER_LINEAR)
    a = np.pad(a, ((g.pad_top, g.net_h - g.h_net - g.pad_top), (g.pad_left, g.net_w - g.w_net - g.pad_left), (0, 0)))
    return (a * (1 - g.lo) + g.lo).transpose(2, 0, 1)[None]


def check_warp(H: np.ndarray, img: np.ndarray, g: Geo = ACT_GEO) -> float:
    """The GridSample graph vs OpenCV's warpPerspective on a real top picture, both as network inputs ([0,1] scale)."""
    import cv2

    x = ops.parameter([1, 3, g.net_h, g.net_w], np.float32)
    got = ov.Core().compile_model(ov.Model([grid_op(x.output(0), H, g)], [x]), "CPU")(net_input(img, g))[0][0]
    want = net_input(cv2.warpPerspective(img, H, (CAM_W, CAM_H)), g)[0]
    keep = np.zeros((g.net_h, g.net_w), bool)
    keep[g.pad_top + 6:g.pad_top + g.h_net - 6, g.pad_left + 6:g.pad_left + g.w_net - 6] = True
    keep &= (warp_grid(H, g)[0, ..., 0] > -2)  # skip the black outside part (OpenCV's border differs by a pixel)
    return float(np.abs(got[:, keep] - want[:, keep]).mean() / (1 - g.lo))


def preview(rec: dict, H: np.ndarray, live_img: np.ndarray) -> None:
    """logs/top_warp_<model>.jpg: training picture | live picture | live warped (training picture ghosted on it)."""
    import cv2

    tr = cv2.imread(str(export_dir(rec) / "top_ref.png"))
    if tr is None:
        return
    w = cv2.warpPerspective(live_img, H, (CAM_W, CAM_H))
    ghost = cv2.addWeighted(w, 0.6, tr, 0.4, 0)
    tiles = [tr.copy(), live_img.copy(), ghost]
    for im, t in zip(tiles, ("training view", "live now", "live, corrected (+ghost)")):
        cv2.putText(im, t, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    out = ROOT / "logs" / f"top_warp_{rec['name']}.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), np.hstack(tiles))
    print(f"  preview: {out}")


def warp_slot(x: ov.Output, H: np.ndarray, g: Geo) -> ov.Node:
    """Stacked cameras [n, 1, 3, h, w] (SmolVLA): warp the top camera's slot, pass the others through."""
    n = x.get_partial_shape()[0].get_length()
    parts = ops.split(x, ops.constant(np.int64(0)), n)
    outs = []
    for i in range(n):
        p = parts.output(i)
        if i == g.slot:
            p4 = ops.reshape(p, ops.constant(np.array([1, 3, g.net_h, g.net_w], np.int64)), False)
            p = ops.reshape(grid_op(p4.output(0), H, g), ops.constant(np.array([1, 1, 3, g.net_h, g.net_w], np.int64)),
                            False).output(0)
        outs.append(p)
    return ops.concat(outs, 0)


def add_top_warp(m: ov.Model, H: np.ndarray, g: Geo) -> ov.Node:
    want = "images.top" if g.slot is None else "images"
    param = next(p for p in m.get_parameters() if want in p.output(0).get_names())
    users = list(param.output(0).get_target_inputs())
    gs = grid_op(param.output(0), H, g) if g.slot is None else warp_slot(param.output(0), H, g)
    for inp in users:
        inp.replace_source_output(gs.output(0))
    return gs


# ---- paint plate (place) ----------------------------------------------------------------------------------
# The yellow plate stood empty on its spot in every place demo; it is gone from the cell now, and the place
# policies hesitate without it (slow or no grasp at the spot). --paint-plate paints the model's own demo plate
# back into its top picture (after the warp, in the training view the policy sees), inside the plate's demo
# disc only, and only where the live pixel looks like bare table: grey (low colour) and table-bright. The arm,
# a brick, a hand stay as they are; a shadow on the table darkens the painted plate the same way.
PP_GROW = 5.0  # px (training view, 640x480 camera px) added to the plate's demo radius
PP_CHROMA = (0.11, 0.15)  # max - min of RGB ([0,1]): table (painted) below the first, kept above the second
PP_VAL = (0.22, 0.30, 0.78, 0.86)  # brightness (max of RGB): painted between the middle two, kept outside the outer
PP_SHADE = 0.50  # a live table pixel darker than this darkens the painted plate by brightness / this (>= 0.5)
PP_KEEP_GROW = 2  # network px: a kept (not table) region grows by this much (its edge, white parts next to the arm)
PP_PLATE_SEEN = (0.15, 0.30)  # yellow share of the disc: paint below the first, off above the second (plate is back)


def plate_disc(ref: dict, g: Geo = ACT_GEO) -> np.ndarray:
    """[1, 1, h, w] weight: 1 inside the plate's demo disc (+ PP_GROW) in the network input, soft 1-px edge."""
    b = ref["bowls"]["yellow"]
    s = g.w_net / CAM_W
    cx, cy, r = b["c"][0] * s + g.pad_left, b["c"][1] * s + g.pad_top, (b["r"] + PP_GROW) * s
    v, u = np.mgrid[0:g.net_h, 0:g.net_w].astype(np.float64)
    d = np.hypot(u + 0.5 - (cx + 0.5 * s), v + 0.5 - (cy + 0.5 * s))  # pixel centres (camera px -> net px)
    return np.clip(r - d + 0.5, 0, 1).astype(np.float32)[None, None]


def demo_plate_image(rec: dict, ref: dict, g: Geo = ACT_GEO) -> tuple[np.ndarray, dict]:
    """The model's own demo picture with the plate on its spot (top_ref.png: arm at HOME, plate empty), mapped onto
    its training view by its own markers, as a network input [1, 3, h, w]. Checked: plate found where the demos had
    it, and the disc fully yellow (nothing on or over it)."""
    import cv2

    img = cv2.imread(str(export_dir(rec) / "top_ref.png"))
    if img is None:
        raise SystemExit("paint plate: no top_ref.png next to the export (delete top_ref.json and re-run to make it)")
    w = fit_warp(detect(img), ref_corners(ref))
    tv = cv2.warpPerspective(img, np.array(w["H"]), (CAM_W, CAM_H))
    b = find_bowl(tv, "yellow")
    want = ref["bowls"]["yellow"]
    if b is None:
        raise SystemExit("paint plate: no yellow plate in top_ref.png")
    off = float(np.hypot(*(b[0] - np.array(want["c"]))))
    lo, hi = BOWL_HSV["yellow"]
    yel = cv2.inRange(cv2.cvtColor(tv, cv2.COLOR_BGR2HSV), np.array(lo), np.array(hi)) > 0
    v, u = np.mgrid[0:CAM_H, 0:CAM_W]
    inner = np.hypot(u - want["c"][0], v - want["c"][1]) <= want["r"] - 6
    cover = float(yel[inner].mean())
    info = {"source": "top_ref.png", "plate_off_px": round(off, 1), "plate_r": round(b[1], 1),
            "yellow_frac": round(cover, 3), "ref_warp_px": w["moved_px"]}
    if off > 6 or abs(b[1] - want["r"]) > 6 or cover < 0.9:
        raise SystemExit(f"paint plate: the plate in top_ref.png is not clean / not on its demo spot ({info})")
    return net_input(tv, g), info


def paint_plate_op(x: ov.Output, plate: np.ndarray, disc: np.ndarray) -> ov.Node:
    """x [1, 3, h, w] RGB in [0, 1] (training view) -> x with the plate painted where the disc is bare table."""
    f32 = lambda v: ops.constant(np.asarray(v, np.float32))  # noqa: E731
    ch = ops.constant(np.array([1], np.int64))
    mx = ops.reduce_max(x, ch, True)
    chroma = ops.subtract(mx, ops.reduce_min(x, ch, True))

    def ramp(t, lo, hi):  # 0 at lo .. 1 at hi (hi < lo: falling)
        return ops.clamp(ops.multiply(ops.subtract(t, f32(lo)), f32(1.0 / (hi - lo))), 0.0, 1.0)

    table = ops.multiply(ops.multiply(ramp(chroma, PP_CHROMA[1], PP_CHROMA[0]), ramp(mx, PP_VAL[0], PP_VAL[1])),
                         ramp(mx, PP_VAL[3], PP_VAL[2]))
    k = 2 * PP_KEEP_GROW + 1
    keep = ops.max_pool(ops.subtract(f32(1.0), table), [1, 1], [1, 1], [PP_KEEP_GROW] * 2, [PP_KEEP_GROW] * 2,
                        [k, k]).output(0)
    rgb = ops.split(x, ops.constant(np.int64(1)), 3)
    yel = ramp(ops.subtract(ops.minimum(rgb.output(0), rgb.output(1)), rgb.output(2)), 0.25, 0.35)
    seen = ops.reduce_sum(ops.multiply(yel, f32(disc / disc.sum())), ops.constant(np.array([1, 2, 3], np.int64)), True)
    gate = ramp(seen, PP_PLATE_SEEN[1], PP_PLATE_SEEN[0])  # the real plate is in the disc: leave it alone
    a = ops.multiply(ops.multiply(ops.subtract(f32(1.0), keep), f32(disc)), gate)
    shade = ops.clamp(ops.multiply(mx, f32(1.0 / PP_SHADE)), 0.5, 1.0)
    return ops.add(x, ops.multiply(a, ops.subtract(ops.multiply(f32(plate), shade), x)))


def paint_plate_np(x: np.ndarray, plate: np.ndarray, disc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """numpy reference of paint_plate_op: (painted picture, paint weight [h, w])."""
    import cv2

    r = lambda t, lo, hi: np.clip((t - lo) / (hi - lo), 0, 1)  # noqa: E731
    mx, chroma = x[0].max(0), x[0].max(0) - x[0].min(0)
    table = r(chroma, PP_CHROMA[1], PP_CHROMA[0]) * r(mx, PP_VAL[0], PP_VAL[1]) * r(mx, PP_VAL[3], PP_VAL[2])
    k = 2 * PP_KEEP_GROW + 1
    keep = cv2.dilate((1 - table).astype(np.float32), np.ones((k, k), np.uint8), borderType=cv2.BORDER_REPLICATE)
    yel = r(np.minimum(x[0, 0], x[0, 1]) - x[0, 2], 0.25, 0.35)
    seen = float((yel * disc[0, 0]).sum() / disc.sum())
    a = (1 - keep) * disc[0, 0] * r(seen, PP_PLATE_SEEN[1], PP_PLATE_SEEN[0])
    shade = np.clip(mx / PP_SHADE, 0.5, 1.0)
    return x + a * (plate * shade - x), a


def add_paint_plate(m: ov.Model, top: ov.Node | None, plate: np.ndarray, disc: np.ndarray, g: Geo) -> None:
    """Paint after the top-camera warp (top = its op), or straight on images.top if there is none."""
    if g.policy != "act" or g.slot is not None or g.lo != 0.0:
        raise SystemExit("paint plate: only for ACT exports (images.top in [0, 1])")
    src = top.output(0) if top is not None else \
        next(p for p in m.get_parameters() if "images.top" in p.output(0).get_names()).output(0)
    users = list(src.get_target_inputs())
    y = paint_plate_op(src, plate, disc)
    for inp in users:
        inp.replace_source_output(y.output(0))


# ---- hide bowl bricks (place) -----------------------------------------------------------------------------
# The place policies learned with empty bowls; the sorted bricks pile up in them, and a bowl with bricks in it
# makes the policy hesitate. --hide-bowl-bricks takes one live top picture (right before the stage), finds the
# lilac and orange bowls in it, and marks the brick pixels inside them: saturated pixels that are not the bowl's
# colour (+ the part of such a brick that sticks out over the rim), grown by HB_DILATE px. Their empty-bowl look
# P is that same picture with the bricks inpainted (OpenCV Telea): same light, same bowl position, no alignment
# needed. In the graph, straight on the live images.top input (before the warp: the mask is in live camera
# pixels), y = x + a * (P - x) with a = M * gate: the gate is 1 only while the live pixel still looks like it did
# when the mask was made (max RGB difference below HB_GATE[0], 0 above HB_GATE[1]). So the arm, the brick it
# carries and a brick it just dropped stay visible when they pass over a hidden brick; a brick that was taken out
# or moved shows again. Not stored: re-run before each place stage (tools/orchestrator.py --hide-bowl-bricks).
HB_BOWLS = ("lilac", "orange")
HB_R = {"lilac": (70.0, 85.0), "orange": (64.0, 78.0)}  # fitted radius seen in 23 runs: 79-80 / 72-74 px (4 bricks in lilac: 74.5); outside
# this range the bowl is partly covered (the arm, a hand) or misfound, and it is left alone
HB_LIKE = {"lilac": ((110, 0, 0), (162, 159, 255)), "orange": ((4, 70, 0), (24, 255, 255)),
           "yellow": ((18, 70, 0), (34, 255, 255))}  # HSV: what still counts as the bowl's own colour (shading)
HB_SAT = (70, 45)  # a brick pixel: saturation >= the first, brightness >= the second (highlights, shadows: no)
HB_INSET = 4.0     # px inside the bowl's fitted radius: brick pixels found here (seeds)
HB_RIM = 16.0      # px outside it: brick-coloured pixels joined to a seed count too (a brick over the rim)
HB_MIN_AREA = 120  # camera px: smaller specks are noise (a rim highlight; a whole brick is ~700)
HB_SHADOW = (14, 0.82)  # a brick's shadow: within this many px of it, darker than this x the bowl around it
HB_DILATE = 5      # camera px around each brick + shadow (its soft edge)
HB_GATE = (0.12, 0.20)  # max |RGB - the picture the mask was made from| ([0,1]): hidden below, shown above


def find_bricks(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """Brick pixels in the lilac / orange bowls of a live top picture (BGR, 640x480): mask [480, 640] float32
    (0/1) and what was found per bowl."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    like = {c: cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0 for c, (lo, hi) in HB_LIKE.items()}
    sat = (hsv[..., 1] >= HB_SAT[0]) & (hsv[..., 2] >= HB_SAT[1])
    v, u = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    out = np.zeros(img.shape[:2], np.uint8)
    info = {}
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HB_DILATE + 1, 2 * HB_DILATE + 1))
    for c in HB_BOWLS:
        b = find_bowl(img, c)
        if b is None:
            info[c] = {"bowl": None}
            continue
        if not HB_R[c][0] <= b[1] <= HB_R[c][1]:
            info[c] = {"bowl": None, "r": round(b[1], 1), "why": "radius off: covered?"}
            continue
        d = np.hypot(u - b[0][0], v - b[0][1])
        seed = sat & ~like[c] & (d <= b[1] - HB_INSET)
        seed = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # + brick-coloured pixels just outside the bowl (not any bowl's colour), if they touch a seed
        grow = (seed > 0) | (sat & ~like["lilac"] & ~like["orange"] & ~like["yellow"] & (d <= b[1] + HB_RIM))
        n, lab, st, _ = cv2.connectedComponentsWithStats(grow.astype(np.uint8), 8)
        keep = np.zeros(n, bool)
        for i in np.unique(lab[seed > 0]):
            keep[i] = i > 0 and st[i, cv2.CC_STAT_AREA] >= HB_MIN_AREA
        m = keep[lab].astype(np.uint8)
        if m.any():  # + its shadow on the bowl: darker than the bowl's own brightness around it, next to it
            val = hsv[..., 2].astype(np.float32)
            own = (like[c] & (d <= b[1] - HB_INSET) & (m == 0)).astype(np.float32)
            loc = cv2.blur(val * own, (41, 41)) / np.maximum(cv2.blur(own, (41, 41)), 1e-3)
            ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HB_SHADOW[0] + 1, 2 * HB_SHADOW[0] + 1))
            near = cv2.dilate(m, ks) > 0
            m |= (near & (val < HB_SHADOW[1] * loc) & (d <= b[1] + HB_RIM)).astype(np.uint8)
        m = cv2.dilate(m, k) & (d <= b[1] + HB_RIM + HB_DILATE).astype(np.uint8)
        out |= m
        info[c] = {"c": b[0].round(1).tolist(), "r": round(b[1], 1), "bricks": int(keep.sum()),
                   "px": int(m.sum())}
    return out.astype(np.float32), info


def hide_bricks_data(img: np.ndarray, g: Geo = ACT_GEO) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, dict]:
    """(ref, plate, mask) as network inputs from one live top picture, or None if no brick is in a bowl:
    ref = the picture itself [1, 3, h, w], plate = it with the bricks inpainted, mask [1, 1, h, w] in [0, 1]."""
    import cv2

    m, info = find_bricks(img)
    if not m.any():
        return None, info
    plate = cv2.inpaint(img, (m > 0).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    mn = cv2.resize(m, (g.w_net, g.h_net), interpolation=cv2.INTER_AREA)
    mn = np.pad(mn, ((g.pad_top, g.net_h - g.h_net - g.pad_top), (g.pad_left, g.net_w - g.w_net - g.pad_left)))
    mn[mn < 1e-3] = 0.0
    info["mask_net_px"] = int((mn > 0).sum())
    return (net_input(img, g), net_input(plate, g), mn[None, None].astype(np.float32)), info


# Paste mode (used by --hide-bowl-bricks once config/cell_ref/empty_bowls_top.png exists): instead of finding
# the bricks, the WHOLE lilac and orange bowls (+ HB_PASTE_GROW px: rim, shadow, a brick over the rim) are
# replaced by a photo of them empty (--capture-empty-bowls). The photo is mapped into the live view with the ArUco
# markers, then shifted onto each bowl's live centre (a bowl nudged since), and its colour is matched to the live
# table ring around the bowl. Same gate as above, so the arm and the brick it carries still show over the bowl.
HB_REF = ROOT / "config" / "cell_ref" / "empty_bowls_top.png"
HB_PASTE_GROW = 14.0  # px beyond the bowl's fitted radius
HB_FEATHER = 3.0      # px: soft edge of the pasted disc (Gaussian sigma)
HB_GAIN = (0.85, 1.18)  # per-channel colour match of the photo to the live table ring around the bowl (clipped)
HB_SHIFT_MAX = 30.0   # px: a live bowl centre further than this from where the photo puts it is not trusted


def capture_empty_bowls() -> int:
    """--capture-empty-bowls: save the live top picture (median of 5 frames) + its markers + the bowl discs as the
    empty-bowl reference. Bowls empty, the arm at HOME, hands out of the picture."""
    import cv2

    sys.path.insert(0, str(ROOT / "tools"))
    import cam_lock as L

    live = live_top()
    img = np.median(np.stack(L.frames("top", 5, gap=0.05)), 0).astype(np.uint8)
    bowls, bad = {}, []
    for c in HB_BOWLS:
        b = find_bowl(img, c)
        if b is None or not HB_R[c][0] <= b[1] <= HB_R[c][1]:
            bad.append(f"{c} bowl {'not found' if b is None else f'radius {b[1]:.0f} px (covered?)'}")
            continue
        bowls[c] = {"c": [round(float(v), 2) for v in b[0]], "r": round(float(b[1]), 2)}
    if bad:
        raise SystemExit("capture empty bowls: " + "; ".join(bad) + " - nothing saved")
    _, info = find_bricks(img)
    left = {c: v["bricks"] for c, v in info.items() if v.get("bricks")}
    HB_REF.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(HB_REF), img)
    HB_REF.with_suffix(".json").write_text(json.dumps({
        "taken_at": time.strftime("%Y-%m-%d %H:%M:%S"), "bowls": bowls,
        "markers": {str(i): {"corners": m.round(2).tolist()} for i, m in live.items()}}, indent=1) + "\n")
    print(f"saved {HB_REF} (+ .json): markers {sorted(live)}, " +
          ", ".join(f"{c} r {v['r']:.0f} px at ({v['c'][0]:.0f}, {v['c'][1]:.0f})" for c, v in bowls.items()))
    if left:
        print(f"WARNING: something brick-like is still in the bowl(s) {left} - if so, empty them and capture again")
    return 0


def paste_bowls_data(img: np.ndarray, g: Geo = ACT_GEO) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, dict]:
    """(ref, plate, mask) like hide_bricks_data, with plate = the live picture with the empty-bowl photo pasted
    over each bowl and mask = those discs (soft edge)."""
    import cv2

    meta = json.loads(HB_REF.with_suffix(".json").read_text())
    photo = cv2.imread(str(HB_REF))
    live = detect(img)
    w = fit_warp(ref_corners(meta), live)  # photo -> live view
    H = np.array(w["H"])
    v, u = np.mgrid[0:CAM_H, 0:CAM_W]
    plate = img.copy()
    mask = np.zeros((CAM_H, CAM_W), np.float32)
    info: dict = {"mode": "paste", "photo": meta["taken_at"], "camera_vs_photo_px": w["moved_px"],
                  "markers": w["markers"]}
    for c in HB_BOWLS:
        b = meta["bowls"].get(c)
        if b is None:
            info[c] = {"bowl": None, "why": "not in the photo"}
            continue
        p0 = np.array(b["c"], np.float64)
        e = cv2.perspectiveTransform(np.array([[p0, p0 + [b["r"], 0], p0 + [0, b["r"]]]]), H)[0]
        exp = e[0]
        r = float(np.mean([np.linalg.norm(e[1] - exp), np.linalg.norm(e[2] - exp)]))
        now = find_bowl(img, c)
        shift = np.zeros(2)
        why = "photo position (bowl not seen now)"
        if now is not None and HB_R[c][0] <= now[1] <= HB_R[c][1]:
            d = now[0] - exp
            if np.hypot(*d) <= HB_SHIFT_MAX:
                shift, why = d, "live bowl"
            else:
                why = f"photo position (live bowl {np.hypot(*d):.0f} px away: not trusted)"
        T = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], np.float64)
        pasted = cv2.warpPerspective(photo, T @ H, (CAM_W, CAM_H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        cc = exp + shift
        dist = np.hypot(u - cc[0], v - cc[1])
        R = r + HB_PASTE_GROW
        ring = (dist > R + 4) & (dist < R + 18)
        gain = np.ones(3)
        if ring.sum() > 200:
            gain = np.clip(np.median(img[ring].astype(np.float32), 0) /
                           np.maximum(np.median(pasted[ring], 0), 1.0), *HB_GAIN)
        disc = (dist <= R + 2 * HB_FEATHER).astype(np.float32)
        soft = np.clip((R + HB_FEATHER - dist) / (2 * HB_FEATHER), 0, 1).astype(np.float32)
        plate = np.where(disc[..., None] > 0, np.clip(pasted * gain, 0, 255), plate).astype(np.uint8)
        mask = np.maximum(mask, soft)
        info[c] = {"c": cc.round(1).tolist(), "r": round(r, 1), "paste_r": round(R, 1), "centre": why,
                   "shift_px": shift.round(1).tolist(), "gain": np.round(gain[::-1], 3).tolist()}  # gain as RGB
    if not mask.any():
        return None, info
    mn = cv2.resize(mask, (g.w_net, g.h_net), interpolation=cv2.INTER_AREA)
    mn = np.pad(mn, ((g.pad_top, g.net_h - g.h_net - g.pad_top), (g.pad_left, g.net_w - g.w_net - g.pad_left)))
    mn[mn < 1e-3] = 0.0
    info["mask_net_px"] = int((mn > 0).sum())
    try:  # what the policy sees at the stage start, next to the camera (for a look; never fails the tune)
        out = ROOT / "logs" / "hide_bowls.jpg"
        out.parent.mkdir(exist_ok=True)
        seen = (img + mask[..., None] * (plate.astype(np.float32) - img)).astype(np.uint8)
        cv2.imwrite(str(out), np.hstack([img, seen]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    except Exception:  # noqa: BLE001
        pass
    return (net_input(img, g), net_input(plate, g), mn[None, None].astype(np.float32)), info


def hide_bricks_op(x: ov.Output, ref: np.ndarray, plate: np.ndarray, mask: np.ndarray) -> ov.Node:
    """x [1, 3, h, w] RGB in [0, 1] (live view) -> x + mask * gate * (plate - x); gate: x still looks like ref."""
    f32 = lambda v: ops.constant(np.asarray(v, np.float32))  # noqa: E731
    diff = ops.reduce_max(ops.abs(ops.subtract(x, f32(ref))), ops.constant(np.array([1], np.int64)), True)
    gate = ops.clamp(ops.multiply(ops.subtract(f32(HB_GATE[1]), diff), f32(1.0 / (HB_GATE[1] - HB_GATE[0]))), 0.0, 1.0)
    a = ops.multiply(f32(mask), gate)
    return ops.add(x, ops.multiply(a, ops.subtract(f32(plate), x)))


def hide_bricks_np(x: np.ndarray, ref: np.ndarray, plate: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """numpy reference of hide_bricks_op: (picture, hide weight [h, w])."""
    gate = np.clip((HB_GATE[1] - np.abs(x - ref).max(1, keepdims=True)) / (HB_GATE[1] - HB_GATE[0]), 0, 1)
    a = mask * gate
    return x + a * (plate - x), a[0, 0]


def add_hide_bricks(m: ov.Model, hide: tuple[np.ndarray, np.ndarray, np.ndarray], g: Geo) -> None:
    """Straight on the images.top input, in front of everything (the warp, the plate paint) already there."""
    if g.policy != "act" or g.slot is not None or g.lo != 0.0:
        raise SystemExit("hide bowl bricks: only for ACT exports (images.top in [0, 1])")
    src = next(p for p in m.get_parameters() if "images.top" in p.output(0).get_names()).output(0)
    users = list(src.get_target_inputs())
    y = hide_bricks_op(src, *hide)
    for inp in users:
        inp.replace_source_output(y.output(0))


# ---- plate follow (pick) ----------------------------------------------------------------------------------
# act-pick-v4b, 82 demos: grasp pose (state 3 frames before the jaws close) vs the grasped brick's centre in the
# training view, 50 demos with the brick found: joint units per px (x, y). Leave-one-out error vs ignoring the
# position: pan 3.4 vs 7.2, lift 8.4 vs 13.7, elbow 10.5 vs 16.7 (wrist flex 8.6 vs 8.9: no real dependence,
# left out). A plate shift moves its brick with it, so the same slopes apply to the plate's offset.
PF_SLOPE = np.array([[0.241, -0.079], [-0.176, -0.379], [0.208, 0.458], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    np.float32)  # rows: pan, lift, elbow, wrist flex, wrist roll, gripper
PF_FOLLOW = 0.1  # share the policy follows by itself (plate shifted 20 px in its top picture: 0.05-0.15)
PF_GATE = {"grip": (8.5, 11.5), "state_pan": (-45.0, -38.0), "pan": (-45.0, -35.0)}  # ramps (off, fully on)
PF_MAX_PX = 40.0  # demo bricks span ~120 x 120 px, demo plates ~35 x 75 px: beyond this the fit is a guess


def plate_offset(rec: dict, live: dict, live_img: np.ndarray) -> np.ndarray:
    """The yellow plate's centre in the live top picture, mapped onto the model's training view, minus where it
    stood in its demos (median): [dx, dy] px."""
    import cv2

    ref = train_top_ref(rec)
    if "yellow" not in ref.get("bowls", {}):
        raise SystemExit("plate follow: no yellow plate in this model's demos")
    b = find_bowl(live_img, "yellow")
    if b is None:
        raise SystemExit("plate follow: no yellow plate in the live top picture")
    H = np.array(fit_warp(live, ref_corners(ref))["H"])
    p = cv2.perspectiveTransform(b[0][None, None], H)[0, 0]
    return p - np.array(ref["bowls"]["yellow"]["c"])


def pf_gate(act: np.ndarray, state: np.ndarray) -> np.ndarray:
    """The plate-follow gate per chunk step (numpy reference of plate_follow_op): act [1, T, 6], state [1, 6]."""
    r = lambda x, lo, hi: np.clip((x - lo) / (hi - lo), 0, 1)  # noqa: E731
    opened = np.maximum(np.maximum.accumulate(act[0, :, GRIPPER]), state[0, GRIPPER])
    return r(opened, *PF_GATE["grip"]) * r(state[0, 0], *PF_GATE["state_pan"]) * r(act[0, :, 0], *PF_GATE["pan"])


def plate_follow_op(act: ov.Output, state: ov.Output, delta: np.ndarray) -> ov.Node:
    """act [1, T, 6] (joint units) + gate[step] * delta. Gate: the gripper has been opened (state, or this or an
    earlier step of the chunk) * the arm is on the plate side now (state pan) * this step is (its pan)."""
    T = act.get_partial_shape()[1].get_length()
    f32 = lambda v: ops.constant(np.asarray(v, np.float32))  # noqa: E731

    def ramp(x, lo, hi):
        return ops.clamp(ops.multiply(ops.subtract(x, f32(lo)), f32(1.0 / (hi - lo))), 0.0, 1.0)

    def joint(x, i, axis):
        return ops.gather(x, ops.constant(np.array([i], np.int64)), ops.constant(np.int64(axis)))

    tril = np.tril(np.ones((T, T), np.float32))[None]  # [1, k, j]: step j is at or before step k
    g_row = ops.reshape(joint(act, GRIPPER, 2), ops.constant(np.array([1, 1, T], np.int64)), False)
    seen = ops.reduce_max(ops.add(ops.multiply(g_row, f32(tril)), f32((tril - 1) * 1e4)),
                          ops.constant(np.array([2], np.int64)), True)  # [1, T, 1]: max gripper up to each step
    shape1 = ops.constant(np.array([1, 1, 1], np.int64))
    s_grip = ops.reshape(joint(state, GRIPPER, 1), shape1, False)
    s_pan = ops.reshape(joint(state, 0, 1), shape1, False)
    gate = ops.multiply(ops.multiply(ramp(ops.maximum(seen, s_grip), *PF_GATE["grip"]),
                                     ramp(s_pan, *PF_GATE["state_pan"])), ramp(joint(act, 0, 2), *PF_GATE["pan"]))
    return ops.add(act, ops.multiply(gate, f32(np.asarray(delta, np.float32)[None, None])))


def grip_graph(src: ov.Output, grip: dict, g: Geo) -> ov.Node:
    """The gripper map on the graph's action output (de-normalised around it if the graph outputs normalised actions)."""
    args = (grip["grip_gain"], grip["grip_from"], grip["grip_max"])
    if g.act_norm is None:
        return grip_map(src, *args)
    mean, std = (ops.constant(a) for a in g.act_norm)
    raw = ops.add(ops.multiply(src, std), mean)
    return ops.divide(ops.subtract(grip_map(raw.output(0), *args), mean), std)


# ---- build + check ----------------------------------------------------------------------------------------
def tune(src_xml: Path, grip: dict, top_h: np.ndarray | None, g: Geo = ACT_GEO,
         pf: np.ndarray | None = None, paint: tuple[np.ndarray, np.ndarray] | None = None,
         hide: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> ov.Model:
    m = ov.Core().read_model(src_xml)
    top = None
    if top_h is not None:
        top = add_top_warp(m, top_h, g)
    if paint is not None:  # (plate picture, disc) after the warp
        add_paint_plate(m, top, *paint, g)
    if hide is not None:  # (ref, plate, mask) in front of the warp: live camera pixels
        add_hide_bricks(m, hide, g)
    res = m.outputs[0].get_node()  # the Result node
    src = res.input_value(0)
    if pf is None:
        y = grip_graph(src, grip, g)
    else:  # plate follow (joint units, before the gripper map; it leaves the gripper alone)
        if g.policy != "act" or g.act_norm is not None:
            raise SystemExit("plate follow: only for ACT exports with actions in joint units")
        state = next(p for p in m.get_parameters() if "state" in p.output(0).get_names())
        y = grip_graph(plate_follow_op(src, state.output(0), pf).output(0), grip, g)
    names = src.get_names()
    src.get_tensor().set_names(set())
    y.output(0).get_tensor().set_names(names)  # the output keeps its name ("action")
    res.input(0).replace_source_output(y.output(0))
    m.validate_nodes_and_infer_types()
    return m


def sample_inputs(model: ov.Model, rng: np.random.Generator, g: Geo = ACT_GEO) -> dict[str, np.ndarray]:
    """Random inputs in the graph's own shapes and types (ACT: a state near HOME; SmolVLA: normalised values,
    some prompt tokens, all masks on)."""
    home = json.loads((ROOT / "config" / "home_pose.json").read_text())["home"]
    out = {}
    for p in model.inputs:
        shape = [d.get_length() for d in p.get_partial_shape()]
        name = p.get_any_name()
        dt = p.get_element_type().to_dtype()
        if name == "state" and g.act_norm is None:
            out[name] = np.array([home], np.float32) + rng.normal(0, 5, (1, len(home))).astype(np.float32)
        elif dt == np.bool_ or "mask" in name:
            out[name] = np.ones(shape, dt)
        elif np.issubdtype(dt, np.integer):
            out[name] = rng.integers(1, 1000, shape).astype(dt)
        elif name.startswith("images"):
            out[name] = (rng.random(shape, dtype=np.float32) * (1 - g.lo) + g.lo).astype(np.float32)
        else:
            out[name] = rng.normal(0, 1, shape).astype(np.float32)
    return out


def check_plate_follow(orig_xml: Path, new_xml: Path, inp: dict, inp_a: dict, grip: dict, top_h: np.ndarray | None,
                       g: Geo, pf: np.ndarray) -> None:
    """With d = 0 the graph equals the plain tune; with d != 0 the arm joints move by exactly gate * delta
    (numpy reference), and not at all with the arm at the spot or on its way back with the gripper shut."""
    core = ov.Core()
    a = core.compile_model(core.read_model(orig_xml), "CPU")
    b = core.compile_model(core.read_model(new_xml), "CPU")
    plain = core.compile_model(tune(orig_xml, grip, top_h, g), "CPU")
    zero = core.compile_model(tune(orig_xml, grip, top_h, g, np.zeros(6, np.float32)), "CPU")
    home = json.loads((ROOT / "config" / "home_pose.json").read_text())["home"]
    states = {"HOME": home, "approach (open)": [-18, 0, 30, 60, -20, 30], "holding brick": [-20, -5, 30, 60, -20, 15],
              "at the spot": [-63, 55, -60, 85, -5, 20], "returning": [-30, -30, -40, 97, -35, 5]}
    worst_zero, ok = 0.0, True
    rows = []
    for nm, s in states.items():
        st = np.array([s], np.float32)
        ya = a({**inp_a, "state": st})[0]
        yb = b({**inp, "state": st})[0]
        yp = plain({**inp, "state": st})[0]
        worst_zero = max(worst_zero, float(np.abs(zero({**inp, "state": st})[0] - yp).max()))
        gate = pf_gate(ya, st)
        want = ya[..., :GRIPPER] + gate[None, :, None] * pf[None, None, :GRIPPER]
        ok &= bool(np.allclose(yb[..., :GRIPPER], want, atol=1e-2))
        ok &= bool(np.allclose(yb[..., GRIPPER], yp[..., GRIPPER], atol=1e-4))  # gripper untouched
        moved = float(np.abs(yb[..., :GRIPPER] - yp[..., :GRIPPER]).max())
        rows.append(f"{nm} {gate.max():.2f}/{moved:.1f}")
        if nm in ("at the spot", "returning") and gate.max() == 0 and moved > 1e-3:
            ok = False
    print(f"  plate follow check: d=0 vs plain tune max diff {worst_zero:.2e}; gate max / arm change per state: "
          + ", ".join(rows))
    if worst_zero > 1e-4 or not ok:
        raise SystemExit("plate follow graph does not match the expected result; not applied")


def demo_frames(rec: dict, ref: dict, n: int = 3, plate_px: float | None = 4.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """(top in the training view, wrist) network inputs at the start of n demos spread over the model's training
    set (arm at HOME). plate_px: only demos with the plate within this of its median spot (some demos had it
    ~25 px off)."""
    import cv2
    import pandas as pd

    snap = snapshot_dir(rec)
    eps = pd.concat([pd.read_parquet(f) for f in sorted((snap / "meta" / "episodes").rglob("*.parquet"))])
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    out = []
    for i in np.unique(np.linspace(0, len(eps) - 1, 3 * n).round().astype(int)):
        if len(out) >= n:
            break
        e, ims = eps.iloc[i], []
        for cam in (TOP, "observation.images.wrist"):
            v = snap / "videos" / cam / f"chunk-{int(e[f'videos/{cam}/chunk_index']):03d}" / \
                f"file-{int(e[f'videos/{cam}/file_index']):03d}.mp4"
            ims.append(video_frame(v, float(e[f"videos/{cam}/from_timestamp"]) + 0.3))
        if ims[0] is None or ims[1] is None:
            continue
        try:
            H = np.array(fit_warp(detect(ims[0]), ref_corners(ref))["H"])
        except SystemExit:
            continue
        if plate_px is not None:
            b = find_bowl(cv2.warpPerspective(ims[0], H, (CAM_W, CAM_H)), "yellow")
            if b is None or np.hypot(*(b[0] - np.array(ref["bowls"]["yellow"]["c"]))) > plate_px:
                continue
        out.append((net_input(cv2.warpPerspective(ims[0], H, (CAM_W, CAM_H)), ACT_GEO),
                    net_input(cv2.resize(ims[1], (CAM_W, CAM_H)), ACT_GEO)))
    return out


def check_paint_plate(orig_xml: Path, grip: dict, top_h: np.ndarray | None, g: Geo,
                      paint: tuple[np.ndarray, np.ndarray], demos: list, extra: list[np.ndarray]) -> None:
    """The paint graph vs its numpy reference on real pictures; nothing changes outside the plate's disc; and on
    demo pictures (the plate on its spot) the painted model predicts nearly the same actions as the plain one."""
    core = ov.Core()
    plate, disc = paint
    x = ops.parameter([1, 3, g.net_h, g.net_w], np.float32)
    op = core.compile_model(ov.Model([paint_plate_op(x.output(0), plate, disc)], [x]), "CPU")
    worst_ref = worst_out = 0.0
    outside = disc[0, 0] == 0
    pics = [plate] + [t for t, _ in demos] + list(extra) + [np.random.default_rng(1).random(plate.shape, np.float32)]
    for p in pics:
        got = op(p)[0]
        worst_ref = max(worst_ref, float(np.abs(got - paint_plate_np(p, plate, disc)[0]).max()))
        worst_out = max(worst_out, float(np.abs(got - p)[..., outside].max()))
    plain = core.compile_model(tune(orig_xml, grip, None, g), "CPU")  # no warp: demo pictures are in the training view
    painted = core.compile_model(tune(orig_xml, grip, None, g, paint=paint), "CPU")
    home = np.array([json.loads((ROOT / "config" / "home_pose.json").read_text())["home"]], np.float32)
    worst_act = 0.0
    for top, wrist in demos:
        inp = {"state": home, "images.top": top, "images.wrist": wrist}
        worst_act = max(worst_act, float(np.abs(painted(inp)[0][0, :30] - plain(inp)[0][0, :30]).max()))
    print(f"  paint plate check: graph vs numpy max diff {worst_ref:.1e}, change outside the disc {worst_out:.1e}, "
          f"plate present ({len(demos)} demo pictures): first 30 actions move max {worst_act:.2f}")
    if worst_ref > 2e-3 or worst_out > 0 or not demos or worst_act > 1.0:
        raise SystemExit("paint plate graph does not match the expected result; not applied")


def check_hide_bricks(hide: tuple[np.ndarray, np.ndarray, np.ndarray], g: Geo = ACT_GEO) -> dict:
    """The hide graph vs its numpy reference; nothing changes where the mask is 0; with an empty mask nothing
    changes at all; on the picture the mask was made from the bricks are fully replaced; a moved-in object
    (here: noise patches over the mask) is left alone."""
    core = ov.Core()
    ref, plate, mask = hide
    x = ops.parameter([1, 3, g.net_h, g.net_w], np.float32)
    op = core.compile_model(ov.Model([hide_bricks_op(x.output(0), ref, plate, mask)], [x]), "CPU")
    empty = core.compile_model(ov.Model([hide_bricks_op(x.output(0), ref, plate, np.zeros_like(mask))], [x]), "CPU")
    rng = np.random.default_rng(2)
    arm = ref.copy()  # something new over the bricks: every 2nd 8-px block of the mask's box replaced by noise
    ys, xs = np.nonzero(mask[0, 0])
    blk = np.zeros(mask.shape[2:], bool)
    blk[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    v, u = np.mgrid[0:g.net_h, 0:g.net_w]
    blk &= ((v // 8 + u // 8) % 2 == 0)
    arm[..., blk] = rng.random((3, int(blk.sum())), np.float32)
    pics = [ref, plate, arm, np.clip(ref + rng.normal(0, 0.02, ref.shape), 0, 1).astype(np.float32),
            rng.random(ref.shape, np.float32), np.zeros_like(ref)]
    outside = mask[0, 0] == 0
    worst_ref = worst_out = worst_empty = 0.0
    for p in pics:
        got = op(p)[0]
        worst_ref = max(worst_ref, float(np.abs(got - hide_bricks_np(p, ref, plate, mask)[0]).max()))
        worst_out = max(worst_out, float(np.abs(got - p)[..., outside].max()))
        worst_empty = max(worst_empty, float(np.abs(empty(p)[0] - p).max()))
    on = mask[0, 0] == 1.0
    full = float(np.abs(op(ref)[0] - plate)[..., on].max(initial=0.0))  # bake picture: bricks fully replaced
    new = blk & (mask[0, 0] > 0) & (np.abs(arm - ref).max(1)[0] > HB_GATE[1])
    kept = float(np.abs(op(arm)[0] - arm)[..., new].max(initial=0.0))
    res = {"graph_vs_numpy": worst_ref, "outside_mask": worst_out, "empty_mask": worst_empty,
           "bake_picture_vs_plate": full, "new_object_change": kept}
    print(f"  hide bricks check: graph vs numpy max diff {worst_ref:.1e}, change outside the mask {worst_out:.1e}, "
          f"empty mask change {worst_empty:.1e}; bake picture -> empty bowl (max diff {full:.1e}); "
          f"something new over the bricks kept (change {kept:.1e})")
    if worst_ref > 1e-4 or worst_out > 0 or worst_empty > 0 or full > 1e-5 or not on.any() or kept > 0:
        raise SystemExit("hide bricks graph does not match the expected result; not applied")
    return res


def check(orig_xml: Path, new_xml: Path, grip: dict, top_h: np.ndarray | None, g: Geo = ACT_GEO,
          pf: np.ndarray | None = None, paint: tuple | None = None, paint_pics: tuple = ((), ()),
          hide: tuple | None = None, gpu: bool = True) -> None:
    core = ov.Core()
    gain, grip_from, grip_max = grip["grip_gain"], grip["grip_from"], grip["grip_max"]
    # the gripper map itself, on a ramp of gripper commands 0..70
    x = ops.parameter([1, 71, 6], np.float32)
    ramp = np.zeros((1, 71, 6), np.float32)
    ramp[..., GRIPPER] = np.arange(71)
    ramp[..., :GRIPPER] = np.arange(GRIPPER) * 10 - 20
    got = core.compile_model(ov.Model([grip_map(x.output(0), gain, grip_from, grip_max)], [x]), "CPU")(ramp)[0]
    gr = ramp[..., GRIPPER]
    want = np.minimum(gr + (gain - 1) * np.maximum(gr - grip_from, 0), grip_max)
    if not (np.allclose(got[..., GRIPPER], want) and np.allclose(got[..., :GRIPPER], ramp[..., :GRIPPER])):
        raise SystemExit("gripper map is wrong")
    print("  gripper map: " + ", ".join(f"{int(v)}->{got[0, int(v), GRIPPER]:.0f}" for v in (5, 12, 20, 26, 30, 40, 55)))
    a, b = core.read_model(orig_xml), core.read_model(new_xml)
    assert b.outputs[0].get_any_name() == a.outputs[0].get_any_name(), "output name changed"
    inp = sample_inputs(a, np.random.default_rng(0), g)
    if hide is not None:  # the live picture the bricks were found in: the hide is fully on there
        inp["images.top"] = hide[0].copy()
    inp_a = dict(inp)
    if hide is not None:  # the original graph fed with the hidden (then warped, painted) top image
        x = ops.parameter([1, 3, g.net_h, g.net_w], np.float32)
        inp_a["images.top"] = core.compile_model(ov.Model([hide_bricks_op(x.output(0), *hide)], [x]),
                                                 "CPU")(inp["images.top"])[0]
    if top_h is not None:  # the original graph fed with the warped top image must give the same actions
        x = ops.parameter([1, 3, g.net_h, g.net_w], np.float32)
        warp = core.compile_model(ov.Model([grid_op(x.output(0), top_h, g)], [x]), "CPU")
        if g.slot is None:
            inp_a["images.top"] = warp(inp_a["images.top"])[0]
        else:
            ims = inp["images"].copy()
            ims[g.slot] = warp(ims[g.slot])[0]
            inp_a["images"] = ims
    if paint is not None:  # ... and then painted
        x = ops.parameter([1, 3, g.net_h, g.net_w], np.float32)
        inp_a["images.top"] = core.compile_model(ov.Model([paint_plate_op(x.output(0), *paint)], [x]),
                                                 "CPU")(inp_a["images.top"])[0]
    ya = core.compile_model(a, "CPU")(inp_a)[0]
    del a
    yb = core.compile_model(b, "CPU")(inp)[0]
    if g.act_norm is not None:  # compare in joint units
        ya, yb = ya * g.act_norm[1] + g.act_norm[0], yb * g.act_norm[1] + g.act_norm[0]
    gr = ya[..., GRIPPER]
    want = np.minimum(gr + (gain - 1) * np.maximum(gr - grip_from, 0), grip_max)
    arm_want = ya[..., :GRIPPER]
    if pf is not None:  # + the plate follow shift where its gate is on
        arm_want = arm_want + pf_gate(ya, inp["state"])[None, :, None] * pf[None, None, :GRIPPER]
    arm_ok = np.allclose(arm_want, yb[..., :GRIPPER], atol=1e-2)
    grip_ok = np.allclose(yb[..., GRIPPER], want, atol=2e-2)
    print(f"  check on CPU: arm joints as expected {arm_ok}, gripper mapped {grip_ok}")
    if not (arm_ok and grip_ok):
        raise SystemExit("tuned graph does not match the expected result; not applied")
    if pf is not None:
        check_plate_follow(orig_xml, new_xml, inp, inp_a, grip, top_h, g, pf)
    if paint is not None:
        check_paint_plate(orig_xml, grip, top_h, g, paint, *paint_pics)
    if hide is not None:
        check_hide_bricks(hide, g)
    if not gpu:  # --hide-bowl-bricks re-makes it before every place stage: no extra GPU compile (Studio does one)
        return
    try:
        cg = core.compile_model(b, "GPU")
        cg(inp)
        t = time.perf_counter()
        for _ in range(20):
            cg(inp)
        print(f"  GPU latency {(time.perf_counter() - t) / 20 * 1000:.1f} ms per chunk")
    except Exception as e:  # noqa: BLE001 - the latency is a nice-to-have
        print(f"  GPU latency check skipped: {e}")


def describe_ref(ref: dict) -> str:
    parts = []
    for i, m in sorted(ref["markers"].items(), key=lambda kv: int(kv[0])):
        d = np.subtract(m["last_ep"], m["first_ep"])
        parts.append(f"m{i} {np.hypot(*d):.0f}/{m['spread_p90_px']:.0f}")
    return (f"{ref['episodes']} demos ({ref['source'][9:25]}), camera drift during them per marker "
            f"(first->last demo / p90 px): " + " ".join(parts))


def graph_files(d: Path, g: Geo) -> tuple[Path, ...]:
    """(graph xml, bin, backup of the trained xml, its bin, temp xml, tune.json) of an export."""
    s = Path(g.xml).stem
    return (d / f"{s}.xml", d / f"{s}.bin", d / f"{s}.orig.xml", d / f"{s}.orig.bin", d / f"{s}.tune.xml",
            d / "tune.json")


def apply(name: str, args, live: dict | None, live_img: np.ndarray | None) -> None:
    import cv2

    rec = model_record(name)
    d = export_dir(rec)
    g = export_geo(d)
    xml, bin_, oxml, obin, tmp, meta = graph_files(d, g)
    if not xml.exists():
        raise SystemExit(f"{name}: no {xml.name} in {d}")
    prev = json.loads(meta.read_text()) if meta.exists() else {}
    grip = {k: getattr(args, k) if getattr(args, k) is not None else prev.get(k, v) for k, v in GRIP_DEFAULT.items()}
    print(f"{name}:")
    pf = None
    if getattr(args, "plate_follow", False) and "pick" not in name:
        print("  plate follow: skipped (not a pick model)")
    elif getattr(args, "plate_follow", False):
        if name != "act-pick-v4b":
            print("  plate follow: NOTE slopes + following share were measured on act-pick-v4b")
        off = plate_offset(rec, live, live_img)
        dist = float(np.hypot(*off))
        print(f"  plate follow: live plate is ({off[0]:+.0f}, {off[1]:+.0f}) px from its demo spot (training view)")
        if dist > PF_MAX_PX:
            raise SystemExit(f"  plate {dist:.0f} px off - beyond what the demos cover ({PF_MAX_PX:.0f} px); "
                             "move the plate back instead; not applied")
        pf = (args.plate_follow_gain * PF_SLOPE @ off).astype(np.float32)
        print(f"  plate follow: grasp shift (gain {args.plate_follow_gain:.2f}) pan {pf[0]:+.1f} lift {pf[1]:+.1f} "
              f"elbow {pf[2]:+.1f} during approach + grasp only")
    paint = paint_info = None
    if getattr(args, "paint_plate", False) and ("pick" in name or pf is not None):
        print("  paint plate: skipped (place models only)")
    elif getattr(args, "paint_plate", False):
        pref = train_top_ref(rec)
        if "yellow" not in pref.get("bowls", {}):
            raise SystemExit("paint plate: no yellow plate in this model's demos")
        plate, paint_info = demo_plate_image(rec, pref, g)
        paint = (plate, plate_disc(pref, g))
        paint_info.update({"c": pref["bowls"]["yellow"]["c"], "r": pref["bowls"]["yellow"]["r"], "grow": PP_GROW,
                           "chroma": PP_CHROMA, "val": PP_VAL, "shade": PP_SHADE, "keep_grow": PP_KEEP_GROW,
                           "plate_seen": PP_PLATE_SEEN})
        print(f"  paint plate: demo plate from {paint_info['source']} (plate {paint_info['plate_off_px']} px from its "
              f"demo median, {100 * paint_info['yellow_frac']:.0f} % yellow), disc r {pref['bowls']['yellow']['r']} "
              f"+ {PP_GROW:.0f} px at {pref['bowls']['yellow']['c']}")
    hide = hide_info = None
    if getattr(args, "hide_bowl_bricks", False) and ("pick" in name or pf is not None):
        print("  hide bowl bricks: skipped (place models only)")
    elif getattr(args, "hide_bowl_bricks", False):
        if live_img is None:
            raise SystemExit("hide bowl bricks: no live top picture")
        mode = getattr(args, "hide_mode", "auto")
        if mode == "paste" and not HB_REF.exists():
            raise SystemExit(f"hide bowl bricks: no empty-bowl photo ({HB_REF}); run --capture-empty-bowls first")
        if mode != "inpaint" and HB_REF.exists():
            hide, hide_info = paste_bowls_data(live_img, g)
            print(f"  hide bowl bricks: empty-bowl photo of {hide_info['photo']} pasted over " +
                  "; ".join(f"{c} (r {v['paste_r']:.0f} px, centre from {v['centre']}, shift {v['shift_px']} px, "
                            f"colour x{v['gain']})" if "paste_r" in v else f"{c}: {v.get('why')}"
                            for c, v in hide_info.items() if c in HB_BOWLS) +
                  f"; camera {hide_info['camera_vs_photo_px']:.0f} px from the photo")
            if hide is None:
                print("  hide bowl bricks: no bowl to paste - nothing hidden")
            mode = "paste"
        else:
            hide, hide_info = hide_bricks_data(live_img, g)
            mode = "inpaint"
        found = ", ".join(f"{c} " + (str(v["bricks"]) + " brick(s)" if v.get("bowl", 1) is not None else
                                     f"left alone (r {v['r']} px: partly covered?)" if "r" in v else "not found")
                          for c, v in hide_info.items() if c in HB_BOWLS and mode == "inpaint")
        if mode == "paste":
            pass
        elif hide is None:
            print(f"  hide bowl bricks: no bricks in the bowls ({found}) - nothing hidden")
        else:
            print(f"  hide bowl bricks: {found}; {hide_info['mask_net_px']} network px shown as empty bowl "
                  f"while they look as now")
    top = None
    if args.top_warp == "auto":
        ref = train_top_ref(rec)
        print(f"  training view: {describe_ref(ref)}")
        top = fit_warp(live, ref_corners(ref))
        print(f"  live camera vs its demos: moved {top['moved_px']:.0f} px; {top['kind']} warp on markers "
              f"{top['markers']}, residual {top['residual_px']:.1f} px, black border {top['black_border_pct']:.1f} %")
        if top["residual_px"] > 8:
            raise SystemExit(f"  the warp does not fit the markers (residual {top['residual_px']:.1f} px); not applied")
        if top["black_border_pct"] > 12:
            print("  WARNING: the camera is far off - a big part of the training view is not in the picture; re-aim it")
        err = check_warp(np.array(top["H"]), live_img, g)
        print(f"  warp graph vs OpenCV: mean difference {err:.4f}")
        if err > 0.03:
            raise SystemExit("  the warp graph does not match OpenCV; not applied")
        top["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        old = prev.get("top_warp")
        if old and all(prev.get(k) == v for k, v in grip.items()) and xml.exists() and oxml.exists() \
                and pf is None and "plate_follow" not in prev and paint is None and "paint_plate" not in prev                 and hide is None and "hide_bricks" not in prev:
            pts = np.array([[[x, y] for x in (0, CAM_W / 2, CAM_W) for y in (0, CAM_H / 2, CAM_H)]], np.float64)
            change = float(np.abs(cv2.perspectiveTransform(pts, np.array(top["H"])) -
                                  cv2.perspectiveTransform(pts, np.array(old["H"]))).max())
            if change < args.min_change:
                print(f"  camera unchanged since {old['measured_at']} ({change:.1f} px) - keeping the current export")
                return
        preview(rec, np.array(top["H"]), live_img)
    elif args.top_warp == "keep":
        top = prev.get("top_warp")
    top_h = np.array(top["H"]) if top else None
    if not oxml.exists():
        shutil.copy2(xml, oxml)
        shutil.copy2(bin_, obin)
        print(f"  kept the trained export as {oxml.name} / {obin.name}")
    m = tune(oxml, grip, top_h, g, pf, paint, hide)
    ov.save_model(m, tmp, compress_to_fp16=False)
    del m
    try:
        pics = ((), ())
        if paint is not None:
            extra = [] if live_img is None or top_h is None else \
                [net_input(cv2.warpPerspective(live_img, top_h, (CAM_W, CAM_H)), g)]
            pics = (demo_frames(rec, train_top_ref(rec)), extra)
        check(oxml, tmp, grip, top_h, g, pf, paint, pics, hide, gpu=not getattr(args, "hide_bowl_bricks", False))
    except BaseException:
        tmp.unlink(missing_ok=True)
        tmp.with_suffix(".bin").unlink(missing_ok=True)
        raise
    tmp.replace(xml)
    tmp.with_suffix(".bin").replace(bin_)
    rec_ = {**grip, "top_warp": top, "tuned_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if hide is not None:
        rec_["hide_bricks"] = {**hide_info, "gate": HB_GATE, "dilate": HB_DILATE}
        print("  bricks in the bowls hidden from the top camera (load the model again for it to take effect)")
    if paint is not None:
        rec_["paint_plate"] = paint_info
        meta.write_text(json.dumps(rec_) + "\n")
        print(f"  tuned: gripper {grip}, top warp {'none' if not top else str(top['moved_px']) + ' px undone'}, "
              f"plate painted in (load the model again for it to take effect)")
        return
    if pf is not None:
        rec_["plate_follow"] = {"offset_px": off.round(1).tolist(), "gain": args.plate_follow_gain,
                                "delta": pf.round(2).tolist(), "gate": PF_GATE}
        meta.write_text(json.dumps(rec_) + "\n")
        print(f"  tuned: gripper {grip}, top warp {'none' if not top else str(top['moved_px']) + ' px undone'}, "
              f"plate follow {pf[:3].round(1).tolist()} (load the model again for it to take effect)")
        return
    meta.write_text(json.dumps(rec_) + "\n")
    print(f"  tuned: gripper {grip}, top warp "
          f"{'none' if not top else str(top['moved_px']) + ' px undone'} (load the model again for it to take effect)")


def camera_report(names: list[str], live: dict) -> None:
    for n in names:
        rec = model_record(n)
        try:
            ref = train_top_ref(rec)
            w = fit_warp(live, ref_corners(ref))
        except (SystemExit, Exception) as e:  # noqa: BLE001 - report the others anyway
            print(f"{n}: {e}")
            continue
        off = {i: np.round(live[i].mean(0) - ref_corners(ref)[i].mean(0)).astype(int).tolist() for i in w["markers"]}
        tj = export_dir(rec) / "tune.json"
        cur = (json.loads(tj.read_text()) if tj.exists() else {}).get("top_warp")
        print(f"{n}: {describe_ref(ref)}\n  live camera is {w['moved_px']:.0f} px from these demos "
              f"(live - demo per marker {off}; black border after warp {w['black_border_pct']} %); applied warp: {'none' if not cur else str(cur['moved_px']) + ' px at ' + cur['measured_at']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="*", help="Studio model names or ids")
    ap.add_argument("--cell", action="store_true", help="the pick/place models named in config/cell.json")
    ap.add_argument("--grip-gain", type=float, help="how much further the gripper opens above --grip-from (1.5)")
    ap.add_argument("--grip-from", type=float, help="commands below this are left alone, brick ~15 wide (12)")
    ap.add_argument("--grip-max", type=float, help="never command the gripper wider than this (60)")
    ap.add_argument("--top-warp", choices=["keep", "auto", "off"], default="keep",
                    help="auto: measure the top camera now and undo its move vs each model's demos; "
                         "keep (default): reuse the stored warp; off: remove it")
    ap.add_argument("--min-change", type=float, default=2.0,
                    help="--top-warp auto: rebuild only if the warp moves the picture by more than this (px)")
    ap.add_argument("--plate-follow", action="store_true",
                    help="pick models: measure the yellow plate in the live top camera and shift the approach + "
                         "grasp by the part of its offset the policy does not follow (off by default; not stored)")
    ap.add_argument("--plate-follow-gain", type=float, default=1 - PF_FOLLOW,
                    help="share of the demo-fit grasp shift to add (0.9: what the policy does not follow itself)")
    ap.add_argument("--paint-plate", action="store_true",
                    help="place models: paint the model's empty demo plate back into its top picture, inside the "
                         "plate's demo disc, where the live picture is bare table (off by default; not stored)")
    ap.add_argument("--hide-bowl-bricks", action="store_true",
                    help="place models: take a live top picture, find the bricks lying in the lilac / orange bowls "
                         "and show the policy the empty bowl there while they stay as they are (off by default; "
                         "not stored: re-run before every place stage)")
    ap.add_argument("--hide-mode", choices=["auto", "paste", "inpaint"], default="auto",
                    help="--hide-bowl-bricks: paste = the whole bowls from the empty-bowl photo "
                         "(--capture-empty-bowls), inpaint = only the bricks found now; auto (default): paste "
                         "once the photo exists")
    ap.add_argument("--capture-empty-bowls", action="store_true",
                    help="save the live top picture as the empty-bowl photo for --hide-bowl-bricks (bowls empty, "
                         "arm at HOME, hands out of the picture); no model needed")
    ap.add_argument("--camera-report", action="store_true", help="only report, change nothing")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()
    if args.capture_empty_bowls:
        return capture_empty_bowls()

    names = list(args.models)
    if args.cell or (args.camera_report and not names):
        c = cfg()
        names += [c[k] for k in CELL_KEYS if c.get(k) and c[k] not in names]
    if not names:
        ap.error("name a model, or use --cell")
    if args.show or args.restore:
        for n in names:
            d = export_dir(model_record(n))
            xml, bin_, oxml, obin, _, meta = graph_files(d, export_geo(d))
            if args.show:
                print(f"{n}: {meta.read_text().strip() if meta.exists() else 'trained export (no tune)'}")
            elif oxml.exists():
                shutil.copy2(oxml, xml)
                shutil.copy2(obin, bin_)
                meta.unlink(missing_ok=True)
                print(f"{n}: restored the trained export")
            else:
                print(f"{n}: nothing to restore (never tuned)")
        return 0
    live = live_img = None
    if args.top_warp == "auto" or args.camera_report or args.plate_follow:
        sys.path.insert(0, str(ROOT / "tools"))
        import cam_lock as L

        live = live_top()
        live_img = L.frames("top", 1)[-1]
        print(f"live top camera: markers {sorted(live)}")
    elif args.hide_bowl_bricks:  # one picture is all it needs (median of 3 frames: less noise)
        sys.path.insert(0, str(ROOT / "tools"))
        import cam_lock as L

        if "top" not in L.SHARED:
            L.open_source("top")
        live_img = np.median(np.stack(L.frames("top", 3, gap=0.03)), 0).astype(np.uint8)
    if args.camera_report:
        camera_report(names, live)
        return 0
    failed = []
    for n in names:
        try:
            apply(n, args, live, live_img)
        except SystemExit as e:  # one model's problem must not stop the others
            print(f"  {n}: {e}")
            failed.append(n)
    if failed:
        print(f"NOT updated: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
