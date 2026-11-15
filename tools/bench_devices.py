#!/usr/bin/env python3
"""Latency, throughput and power of the cell's OpenVINO models on the CPU, iGPU and NPU.

    python tools/bench_devices.py               # the models in config/cell.json, on CPU, GPU and NPU
    python tools/bench_devices.py --wait-quiet  # first wait until no training / alignment job is running

Models (from config/cell.json): the ACT pick policy (Studio's OpenVINO export; place_good and place_bad
share its graph, which is checked and noted), the anomaly model, and EfficientAd when it exists
(the NPU-friendly alternative). Every model is reshaped to batch 1 (static shapes, as the NPU needs).

Per model x device: cold compile time (no cache), first inference, steady-state latency with the
LATENCY hint (median, p95, p99), throughput with the THROUGHPUT hint, and, when tools/power_log.py
is running, package / iGPU / platform power and NPU load in the same window, plus energy per
inference above an idle baseline measured at the start.

Writes results/bench-<stamp>.json and .md. Pauses while tools/orchestrator.py runs, so it never
competes with the arm, and records which other busy processes were running during each block.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics as st
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import power_report as pr  # noqa: E402

STUDIO_MODELS = Path.home() / ".local" / "share" / "physicalai" / "models"
HEAVY = ("anomaly_train.py", "frame_align.py", "tools/train", "lerobot_train", "benchmark_app")
ARM = ("tools/orchestrator.py", "tools/cell_live.py")


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


class ArmStarted(Exception):
    pass


def log(msg: str) -> None:
    print(f"[bench {dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def procs() -> list[tuple[int, float, str]]:
    out = subprocess.run(["ps", "-eo", "pid=,pcpu=,args="], capture_output=True, text=True).stdout
    res = []
    for line in out.splitlines():
        p = line.strip().split(None, 2)
        if len(p) == 3 and int(p[0]) != os.getpid():
            res.append((int(p[0]), float(p[1]), p[2]))
    return res


def short(args: str) -> str:
    toks = args.split()
    py = next((t for t in toks if t.endswith(".py")), None)
    return (py or toks[0]).rsplit("/", 1)[-1] + ("" if not py else " " + " ".join(toks[toks.index(py) + 1:toks.index(py) + 3]))


def matching(patterns: tuple[str, ...]) -> list[str]:
    return sorted({short(args) for _, _, args in procs()
                   if any(x in args for x in patterns) and "bench_devices" not in args and not args.startswith(("timeout", "bash", "sh "))})


def busy_now(min_pct: float = 30.0) -> list[str]:
    return [f"{pc:.0f}% {short(args)}"
            for _, pc, args in sorted(procs(), key=lambda x: -x[1]) if pc >= min_pct and "power_log" not in args][:6]


def wait_for_arm() -> None:
    first = True
    while matching(ARM):
        if first:
            log("an arm run (orchestrator) is active: paused until it ends")
            first = False
        time.sleep(5)
    if not first:
        log("arm run ended, continuing")


def studio_model_xml(cfg: dict, name: str) -> Path | None:
    try:
        url = f"{cfg['url']}/api/projects/{cfg['project']}/models"
        items = json.load(urllib.request.urlopen(url, timeout=5))
        items = items if isinstance(items, list) else items.get("items", [])
        mid = next((m["id"] for m in items if m.get("name") == name), None)
    except Exception as e:  # Studio down: fall back to nothing, the caller reports it
        log(f"Studio API unavailable ({e}); pass --act")
        return None
    p = STUDIO_MODELS / str(mid) / "exports" / "openvino" / "act.xml" if mid else None
    return p if p and p.exists() else None


def weight_mix(model: ov.Model) -> dict[str, float]:
    c: Counter = Counter()
    for op in model.get_ops():
        if op.get_type_name() != "Constant":
            continue
        et = op.get_output_element_type(0)
        n = int(np.prod(list(op.get_output_shape(0)))) if len(op.get_output_shape(0)) else 1
        bits = getattr(et, "bitwidth", None) or et.size * 8
        c[et.get_type_name()] += n * bits / 8
    tot = sum(c.values()) or 1
    return {k: round(100 * v / tot, 1) for k, v in c.most_common() if v / tot >= 0.005}


def static_batch1(model: ov.Model) -> None:
    shapes = {}
    for inp in model.inputs:
        ps = inp.get_partial_shape()
        if ps.is_dynamic:
            shapes[inp.any_name] = [1 if d.is_dynamic else d.get_length() for d in ps]
    if shapes:
        model.reshape(shapes)


def make_inputs(model: ov.Model, rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {}
    for inp in model.inputs:
        shape = [d.get_length() for d in inp.get_partial_shape()]
        dtype = inp.get_element_type().to_dtype()
        if np.issubdtype(dtype, np.floating):
            out[inp.any_name] = rng.random(shape).astype(dtype)
        else:  # token ids / masks (SmolVLA): ones = valid tokens and unpadded cameras, not all-padding zeros
            out[inp.any_name] = np.ones(shape, dtype=dtype)
    return out


def safe_prop(obj, key: str):
    try:
        v = obj.get_property(key)
        if hasattr(v, "get_type_name"):  # ov.Type -> "f16"
            return v.get_type_name()
        return str(v) if not isinstance(v, (int, float, str, list)) else v
    except Exception:
        return None


def bench_one(core: ov.Core, model: ov.Model, device: str, inputs: dict, a: argparse.Namespace) -> dict:
    for attempt in range(20):
        try:
            return _bench_one(core, model, device, inputs, a)
        except ArmStarted:
            log(f"  {device}: an arm run started mid-measurement; block discarded, will redo it after the run")
    return {"device": device, "error": "kept being interrupted by arm runs"}


class _ArmWatch:
    def __init__(self) -> None:
        self.next = time.perf_counter() + 1.0

    def check(self) -> None:
        if time.perf_counter() >= self.next:
            self.next = time.perf_counter() + 1.0
            if matching(ARM):
                raise ArmStarted


def _bench_one(core: ov.Core, model: ov.Model, device: str, inputs: dict, a: argparse.Namespace) -> dict:
    res: dict = {"device": device}
    watch = _ArmWatch()
    wait_for_arm()
    res["busy_at_start"] = busy_now()
    res["heavy_jobs"] = matching(HEAVY)
    t = time.perf_counter()
    try:
        cm = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
    except Exception as e:
        res["error"] = f"compile failed: {str(e).strip().splitlines()[0][:200]}"
        log(f"  {device}: {res['error']}")
        return res
    res["compile_s"] = round(time.perf_counter() - t, 2)
    res["precision"] = safe_prop(cm, "INFERENCE_PRECISION_HINT")
    res["execution_devices"] = safe_prop(cm, "EXECUTION_DEVICES")
    res["threads"] = safe_prop(cm, "INFERENCE_NUM_THREADS")
    req = cm.create_infer_request()

    t = time.perf_counter()
    req.infer(inputs)
    res["first_ms"] = round((time.perf_counter() - t) * 1000, 2)
    t_end = time.perf_counter() + a.warmup_s
    while time.perf_counter() < t_end:
        req.infer(inputs)
        watch.check()

    lat = []
    w0 = time.time()
    t_start = time.perf_counter()
    while True:
        t = time.perf_counter()
        req.infer(inputs)
        lat.append((time.perf_counter() - t) * 1000)
        pause = time.perf_counter()
        watch.check()
        t_start += time.perf_counter() - pause  # the check itself is not counted
        el = time.perf_counter() - t_start
        if (el >= a.min_s and len(lat) >= a.min_n) or el >= a.max_s:
            break
    w1 = time.time()
    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))]  # noqa: E731
    res.update(n=len(lat), median_ms=round(st.median(lat), 2), mean_ms=round(st.mean(lat), 2),
               p95_ms=round(q(0.95), 2), p99_ms=round(q(0.99), 2), min_ms=round(lat[0], 2), max_ms=round(lat[-1], 2),
               window=[w0, w1])
    log(f"  {device}: compile {res['compile_s']} s, first {res['first_ms']} ms, median {res['median_ms']} ms, "
        f"p95 {res['p95_ms']} ms (n={len(lat)}, {res['precision']})")
    del req, cm

    if a.throughput:
        wait_for_arm()
        try:
            cm = core.compile_model(model, device, {"PERFORMANCE_HINT": "THROUGHPUT"})
            nreq = int(safe_prop(cm, "OPTIMAL_NUMBER_OF_INFER_REQUESTS") or 1)
            q_ = ov.AsyncInferQueue(cm, nreq)
            done = [0]
            q_.set_callback(lambda *_: done.__setitem__(0, done[0] + 1))
            for _ in range(nreq):
                q_.start_async(inputs)
            q_.wait_all()
            done[0] = 0
            t = time.perf_counter()
            while time.perf_counter() - t < a.tput_s:
                q_.start_async(inputs)
                watch.check()
            q_.wait_all()
            el = time.perf_counter() - t
            res["throughput_fps"] = round(done[0] / el, 1)
            res["throughput_nreq"] = nreq
            log(f"  {device}: throughput {res['throughput_fps']} inferences/s with {nreq} requests")
            del q_, cm
        except ArmStarted:
            raise
        except Exception as e:
            res["throughput_error"] = str(e).strip().splitlines()[0][:200]
    res["busy_at_end"] = busy_now()
    return res


def mj_cell(r: dict) -> str:
    v = r.get("pkg_mJ_per_inference_above_idle")
    if v is not None:
        return f"{v:.2f}"
    return "n/m" if r.get("energy_note") else "–"


def fmt(v, nd: int = 1, unit: str = "") -> str:
    if v is None:
        return "–"
    return f"{v:.{nd}f}{unit}" if isinstance(v, (int, float)) else str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=Path, default=ROOT / "config" / "cell.json")
    ap.add_argument("--devices", default="NPU,GPU,CPU")
    ap.add_argument("--act", type=Path, action="append", help="ACT act.xml to benchmark (default: the pick model from cell.json)")
    ap.add_argument("--anomaly", type=Path, action="append", help="anomaly model.xml (default: cell.json's + efficientad_256)")
    ap.add_argument("--all-act", action="store_true", help="benchmark pick, place_good and place_bad separately")
    ap.add_argument("--min-s", type=float, default=10.0, help="minimum seconds of latency sampling per device")
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--warmup-s", type=float, default=2.0)
    ap.add_argument("--tput-s", type=float, default=8.0)
    ap.add_argument("--no-throughput", dest="throughput", action="store_false")
    ap.add_argument("--idle-s", type=float, default=15.0, help="idle baseline length before the first model (0 = skip)")
    ap.add_argument("--wait-quiet", action="store_true", help="wait until no training / alignment / arm job runs")
    ap.add_argument("--idle-each", type=float, default=0.0,
                    help="seconds of idle baseline right before EACH model x device block (energy above idle uses it)")
    ap.add_argument("--label", action="append", default=[], help="label for the --act models, in the same order")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    stamp = dt.datetime.now().strftime("%m%d-%H%M")
    out = a.out or ROOT / "results" / f"bench-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.cell.read_text())

    if a.wait_quiet:
        last = 0.0
        while (jobs := matching(HEAVY + ARM)):
            if time.time() - last > 120:
                log("waiting for: " + "; ".join(jobs))
                last = time.time()
            time.sleep(10)
        log("laptop is quiet, starting")

    # --- which models
    models: list[dict] = []
    act_paths = a.act or []
    if not act_paths:
        names = ["pick_model"] + (["place_good_model", "place_bad_model"] if a.all_act else [])
        for key in names:
            p = studio_model_xml(cfg, cfg[key])
            if p:
                act_paths.append(p)
                models.append({"label": f"ACT {cfg[key]}", "xml": p, "role": "policy"})
            else:
                log(f"no OpenVINO export found for {cfg[key]}")
    else:
        for k, p in enumerate(act_paths):
            if p.exists():
                lab = a.label[k] if k < len(a.label) else f"{p.stem.upper()} {p.parent.parent.parent.name[:8]}"
                models.append({"label": lab, "xml": p, "role": "policy"})
            else:
                log(f"missing {p}")
    an_paths = a.anomaly or [ROOT / "models" / "anomaly" / cfg["anomaly_model"] / "weights" / "openvino" / "model.xml",
                             ROOT / "models" / "anomaly" / "efficientad_256" / "weights" / "openvino" / "model.xml"]
    for p in an_paths:
        if p.exists():
            models.append({"label": p.parent.parent.parent.name, "xml": p, "role": "anomaly"})
        else:
            log(f"missing {p}")

    core = ov.Core()
    devices = [d for d in a.devices.split(",") if d in core.available_devices]
    sysinfo = {
        "openvino": ov.__version__, "python": platform.python_version(), "kernel": platform.release(),
        "devices": {d: core.get_property(d, "FULL_DEVICE_NAME") for d in devices},
        "npu_driver": None,
    }
    try:
        sysinfo["npu_driver"] = core.get_property("NPU", "NPU_DRIVER_VERSION") if "NPU" in devices else None
    except Exception:
        pass
    pcsv = pr.newest_power_csv()
    power_live = bool(pcsv and time.time() - pcsv.stat().st_mtime < 5)
    log(f"OpenVINO {ov.__version__}, devices {devices}, power log {'live: ' + pcsv.name if power_live else 'NOT running'}")

    report: dict = {"stamp": stamp, "started": dt.datetime.now().isoformat(timespec="seconds"), "system": sysinfo,
                    "cell": {k: cfg.get(k) for k in ("pick_model", "place_good_model", "place_bad_model", "anomaly_model",
                                                     "device", "anomaly_device", "frames", "hz", "execution", "request_threshold")},
                    "power_csv": pcsv.name if power_live else None, "models": []}

    idle_win = None
    if power_live and a.idle_s > 0:
        wait_for_arm()
        log(f"idle baseline: {a.idle_s:.0f} s with the benchmark doing nothing")
        report["idle_busy"] = busy_now()
        w0 = time.time()
        time.sleep(a.idle_s)
        idle_win = (w0, time.time())

    rng = np.random.default_rng(0)
    graphs: dict[str, dict] = {}
    for m in models:
        binf = m["xml"].with_suffix(".bin")
        log(f"{m['label']}  ({binf.stat().st_size / 1e6 if binf.exists() else 0:.0f} MB weights)")
        model = core.read_model(m["xml"])
        static_batch1(model)
        sig = {"inputs": {i.any_name: str(i.get_partial_shape()) for i in model.inputs},
               "outputs": {o.any_name: str(o.get_partial_shape()) for o in model.outputs}, "ops": len(model.get_ops())}
        entry = {"label": m["label"], "role": m["role"], "xml": str(m["xml"]).replace(str(Path.home()), "~"),
                 "bin_mb": round(m["xml"].with_suffix(".bin").stat().st_size / 1e6, 1) if m["xml"].with_suffix(".bin").exists() else None,
                 "weights": weight_mix(model), **sig, "runs": []}
        inputs = make_inputs(model, rng)
        for d in devices:
            iw = None
            if power_live and a.idle_each > 0:  # local idle baseline: same conditions, seconds before this block
                wait_for_arm()
                i0 = time.time()
                time.sleep(a.idle_each)
                iw = (i0, time.time())
            r = bench_one(core, model, d, inputs, a)
            if iw:
                r["idle_window"] = iw
            entry["runs"].append(r)
            json.dump(report | {"models": report["models"] + [entry]}, out.open("w"), indent=1, default=str)  # partial save
        report["models"].append(entry)
        graphs[m["label"]] = sig

    # place_good / place_bad share the pick graph? (only the weights differ)
    if not a.all_act and not a.act:
        same = []
        for key in ("place_good_model", "place_bad_model"):
            p = studio_model_xml(cfg, cfg[key])
            if p:
                mm = core.read_model(p)
                static_batch1(mm)
                s = {"inputs": {i.any_name: str(i.get_partial_shape()) for i in mm.inputs},
                     "outputs": {o.any_name: str(o.get_partial_shape()) for o in mm.outputs}, "ops": len(mm.get_ops())}
                pick = next((g for k, g in graphs.items() if k.startswith("ACT ")), None)
                same.append({"model": cfg[key], "same_graph_as_pick": s == pick})
        report["act_same_graph"] = same

    # --- power join
    if power_live:
        time.sleep(1.5)
        rows = pr.load_power(pcsv)
        idle = pr.window_stats(rows, *idle_win) if idle_win else None
        report["idle"] = idle
        idle_w = idle.get("pkg_W_mean") if idle else None
        for m in report["models"]:
            for r in m["runs"]:
                if "window" not in r:
                    continue
                s = pr.window_stats(rows, *r["window"])
                r["power"] = s
                if s and s.get("pkg_J") is not None and r.get("n"):
                    r["pkg_mJ_per_inference"] = round(1000 * s["pkg_J"] / r["n"], 1)
                    base = idle_w
                    if r.get("idle_window"):
                        li = pr.window_stats(rows, *r["idle_window"])
                        base = li.get("pkg_W_mean") if li else idle_w
                        r["idle_pkg_W"] = base
                    if base is not None:
                        above = 1000 * (s["pkg_J"] - base * s["seconds"]) / r["n"]
                        if above > 0:
                            r["pkg_mJ_per_inference_above_idle"] = round(above, 2)
                        else:  # the model added less than the idle noise: no honest per-inference energy
                            r["pkg_mJ_per_inference_above_idle"] = None
                            r["energy_note"] = f"package {s.get('pkg_W_mean')} W during the run, not above idle {base} W"
    report["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    out.write_text(json.dumps(report, indent=1, default=str))

    # --- markdown
    L = [f"# Device benchmark {report['started']}", "",
         f"OpenVINO {ov.__version__} on {sysinfo['devices'].get('CPU', '?')} / {sysinfo['devices'].get('GPU', '?')} / "
         f"{sysinfo['devices'].get('NPU', '?')}. Batch 1, static shapes, random inputs, LATENCY hint; throughput with the "
         f"THROUGHPUT hint. Compile = cold (no cache).", ""]
    if report.get("idle"):
        i = report["idle"]
        L += [f"Idle baseline (benchmark not running; Studio, cameras and viewers still on): package "
              f"{fmt(i.get('pkg_W_mean'))} W, platform {fmt(i.get('psys_W_mean'))} W, {fmt(i.get('pkg_C_max'), 0)} °C.", ""]
    L += ["| model | weights | device | precision | compile s | first ms | median ms | p95 ms | inf/s (tput) | package W | iGPU W | NPU % | mJ/inf above idle | busy alongside |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in report["models"]:
        wmix = ", ".join(f"{k} {v:.0f}%" for k, v in list(m["weights"].items())[:2])
        for r in m["runs"]:
            p = r.get("power") or {}
            if "error" in r:
                L.append(f"| {m['label']} | {wmix} | {r['device']} | – | – | – | – | – | – | – | – | – | – | {r['error']} |")
                continue
            heavy = ", ".join(r.get("heavy_jobs", [])) or "–"
            L.append(f"| {m['label']} | {wmix} | {r['device']} | {r.get('precision') or '–'} | {fmt(r.get('compile_s'))} | "
                     f"{fmt(r.get('first_ms'))} | {fmt(r.get('median_ms'), 2)} | {fmt(r.get('p95_ms'), 2)} | {fmt(r.get('throughput_fps'))} | "
                     f"{fmt(p.get('pkg_W_mean'))} | {fmt(p.get('uncore_W_mean'))} | {fmt(p.get('npu_pct_mean'), 0)} | "
                     f"{mj_cell(r)} | {heavy} |")
    frames, views = cfg.get("frames", 5), 3
    L += ["", "**What it means for one brick:**"]
    for m in report["models"]:
        ok = [r for r in m["runs"] if r.get("median_ms") is not None]
        if not ok:
            continue
        best = min(ok, key=lambda r: r["median_ms"])
        if m["role"] == "anomaly":
            L.append(f"- {m['label']}: {frames} frames x {views} views = {frames * views} inferences = "
                     f"{best['median_ms'] * frames * views:.0f} ms per brick on {best['device']} (fastest).")
        else:
            hz = cfg.get("hz", 30)
            L.append(f"- {m['label']}: one call returns a 100-step chunk (3.3 s of motion at {hz} Hz); "
                     f"{best['median_ms']:.1f} ms median on {best['device']} = {100 * best['median_ms'] / (1000 / hz):.0f}% of one {1000 / hz:.0f} ms control step.")
    if report.get("act_same_graph"):
        L.append("- place_good / place_bad: " + ", ".join(f"{x['model']} {'same graph as pick' if x['same_graph_as_pick'] else 'DIFFERENT graph'}"
                                                      for x in report["act_same_graph"]) + " (so the pick numbers apply).")
    L += ["", "Numbers are measured on the station laptop by `tools/bench_devices.py`; power from `tools/power_log.py` (RAPL)."]
    md = out.with_suffix(".md")
    md.write_text("\n".join(L) + "\n")
    log(f"saved {rel(out)} and {rel(md)}")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
