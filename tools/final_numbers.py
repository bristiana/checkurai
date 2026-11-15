#!/usr/bin/env python3
"""Fill the submission text with measured numbers, each traced to the file it came from.

    python tools/final_numbers.py --template /tmp/submission/DESCRIPTION.md --out /tmp/submission/DESCRIPTION.final.md
    python tools/final_numbers.py ... --smolvla "retrained with ...; on the arm it ..."   # optional SmolVLA sentence
    python tools/final_numbers.py ... --no-report     # reuse the closed_loop*.json files instead of re-running the report

Steps:
1. runs tools/closed_loop_report.py three times, with --truth results/truth.csv:
   - all closed-loop runs since --since (default 20261114-1950)  -> results/closed_loop.md/.json
   - only the PatchCore-era runs (run.json args.anomaly_model patchcore_*)  -> results/closed_loop_patchcore.md/.json
   - only the supervised-classifier runs (any other anomaly_model, e.g. resnet18_synth_v1)  -> results/closed_loop_classifier.md/.json
2. reads the clean device bench results/bench-1114-2009.json (ACT pick model: median ms per chunk on GPU / NPU / CPU)
3. reads results/int8.json (full INT8 drift, conv-only INT8 drift and size)
4. reads the classifier's validation (results/classifier/resnet18_synth_v1_meta.json), its live thresholds (low / high
   from the classifier runs' cycles.jsonl) and the PatchCore baseline on the same photo set
   (results/anomaly_zoo/table.json, row patchcore_views_b2all_fx_384)
5. replaces the {{placeholders}} in the template; prints a source table (placeholder -> value -> file) to stdout
   and writes it next to the output as <out>.sources.md

No number is typed in by hand. A placeholder whose source is missing stays visible as {{name}} and is listed
as MISSING, so it can't slip into the submission unnoticed. A line holding an accuracy placeholder is dropped
while there are no hand labels for that group in results/truth.csv.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
BENCH = ROOT / "results" / "bench-1114-2009.json"
INT8 = ROOT / "results" / "int8.json"
CL = ROOT / "results" / "closed_loop"                  # all runs (.md / .json)
CL_PC = ROOT / "results" / "closed_loop_patchcore"     # PatchCore-era runs
CL_CLS = ROOT / "results" / "closed_loop_classifier"   # supervised classifier runs
CLS_META = ROOT / "results" / "classifier" / "resnet18_synth_v1_meta.json"
ZOO = ROOT / "results" / "anomaly_zoo" / "table.json"
ZOO_BASELINE = "patchcore_views_b2all_fx_384"          # PatchCore on the same 35-good / 24-defect photo set
RUNS = ROOT / "data" / "cell_runs"
HW = re.compile(r"no frame|Sync read failed|status packet|CaptureTimeout|could not open port|no publisher", re.I)

SMOLVLA_DEFAULT = "retrains with both fixes (on a Verda H100 and an A100) were still training at submission, so the cell runs ACT"


def fmt(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}"


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def fail_breakdown(rows: list[dict], src: str) -> tuple[str, str]:
    """Why the unsorted cycles failed: hardware (camera / servo bus), policy (pick timeout / stall), start state."""
    hw = pol = start = op = 0
    for r in rows:
        if r.get("result") == "sorted":
            continue
        res = str(r.get("result", ""))
        if "not at HOME" in res:
            start += 1
            continue
        text = ""
        try:
            for line in (RUNS / r["run"] / "cycles.jsonl").read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("cycle") == r.get("cycle"):
                    text = line
        except OSError:
            pass
        if HW.search(text):
            hw += 1
        elif '"operator"' in text:
            op += 1
        else:
            pol += 1
    parts = []
    if pol:
        parts.append(f"{pol} pick timed out")
    if op:
        parts.append(f"{op} stopped by the operator")
    if hw:
        parts.append(f"{hw} camera or servo-bus fault{'s' if hw > 1 else ''}")
    if start:
        parts.append(f"{start} not at HOME at the start")
    return "; ".join(parts) or "none failed", f"{src} rows + data/cell_runs/*/cycles.jsonl (stage details)"


def runs_by_inspector(since: str) -> dict[str, list[str]]:
    """Run folders since `since`, grouped by the inspection model named in their run.json."""
    out: dict[str, list[str]] = {"patchcore": [], "classifier": []}
    for d in sorted(RUNS.glob("2026*")):
        if d.name < since:
            continue
        try:
            m = str(json.loads((d / "run.json").read_text())["args"].get("anomaly_model") or "")
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        if m:
            out["patchcore" if m.startswith("patchcore") else "classifier"].append(rel(d))
    return out


def run_report(runs: list[str] | None, out: Path, since: str, truth: str) -> None:
    cmd = [PY, str(ROOT / "tools" / "closed_loop_report.py"), "--out", str(out)]
    cmd += ["--since", since] if runs is None else runs
    if (ROOT / truth).exists():
        cmd += ["--truth", truth]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def group_vals(tag: str, path: Path, truth: str) -> dict[str, tuple[str, str]]:
    """Closed-loop numbers of one report (all / pc / cls)."""
    f = path.with_suffix(".json")
    if not f.exists():
        return {}
    g = json.loads(f.read_text())
    src = rel(f)
    p = "" if tag == "all" else tag + "_"
    v: dict[str, tuple[str, str]] = {
        p + "cycles": (str(g["cycles"]), src),
        p + "sorted": (str(g["sorted"]), src),
        p + "fail_breakdown": fail_breakdown(g.get("rows", []), src),
    }
    if g.get("cycle_s", {}).get("p50") is not None:
        v[p + "cycle_p50"] = (fmt(g["cycle_s"]["p50"]), src)
    ms = g.get("infer_ms_per_frame_p50")
    if ms is not None:
        v[p + "infer"] = (f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.1f} ms", src + " (p50, one frame of 3 views)")
    if g.get("labelled"):
        v[p + "accuracy"] = (f"{g['accuracy'] * 100:.0f} %", f"{src} + {truth}")
        v[p + "labelled"] = (str(g["labelled"]), truth)
        v[p + "correct"] = (str(round(g["accuracy"] * g["labelled"])), f"{src} + {truth}")
    e = g.get("energy") or {}
    if e.get("psys_Wh_per_brick_median") is not None:
        es = f"{src} <- {e.get('source')}"
        v[p + "wh_per_brick"] = (fmt(e["psys_Wh_per_brick_median"], 2), es)
        v[p + "w_mean"] = (fmt(e["psys_W_mean_median"]), es)
        v[p + "pkg_c_max"] = (fmt(e["pkg_C_max"], 0), es)
    return v


def cls_thresholds(runs: list[str]) -> tuple[float, float, str] | None:
    """The good limit and the sure-defect line the classifier used live (from the last run that logged them)."""
    for r in reversed(runs):
        try:
            lines = (ROOT / r / "cycles.jsonl").read_text().splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                i = json.loads(line).get("inspection") or {}
            except json.JSONDecodeError:
                continue
            if i.get("low") is not None and i.get("high") is not None:
                return float(i["low"]), float(i["high"]), f"{r}/cycles.jsonl (inspection.low / .high)"
    return None


def classifier_photo_vals() -> dict[str, tuple[str, str]]:
    v: dict[str, tuple[str, str]] = {}
    if CLS_META.exists():
        m = json.loads(CLS_META.read_text())
        val = m["val"]
        rec = val["recall_at_thr"]
        caught = sum(int(x.split("/")[0]) for x in rec.values())
        total = sum(int(x.split("/")[1]) for x in rec.values())
        s = rel(CLS_META) + " (val; threshold rule: " + str(m.get("threshold_rule")) + ")"
        v["cls_auroc"] = (fmt(val["auroc"], 2), s)
        v["cls_recall"] = (f"{caught}/{total}", s)
        v["cls_fa"] = (f"{val['goods_fa_at_thr']}/{val['goods']}", s)
        v["cls_smalldot"] = (rec.get("marker_smalldot", "?"), s)
    if ZOO.exists():
        for row in json.loads(ZOO.read_text()):
            if row.get("name") == ZOO_BASELINE:
                s = f"{rel(ZOO)} ({ZOO_BASELINE})"
                v["pc_auroc"] = (fmt(row["auroc"], 2), s)
                v["pc_recall"] = (str(row["recall"]), s)
                v["pc_fa"] = (str(row.get("false_alarms_2fold") or row.get("false_alarms")), s + " 2-fold false alarms")
                if row.get("ms_3views_eval") is not None:
                    v["pc_ms_photo"] = (fmt(row["ms_3views_eval"], 0), s + f" ms for 3 views on {row.get('eval_device')}")
    return v


def act_bench() -> dict[str, tuple[float, str]]:
    d = json.loads(BENCH.read_text())
    out = {}
    for m in d["models"]:
        if m.get("label") != "ACT act-pick-v6-h100":
            continue
        for run in m.get("runs", []):
            if run.get("median_ms") is not None:
                out[run["device"]] = (float(run["median_ms"]), rel(BENCH))
    return out


def int8_line() -> tuple[str, str]:
    d = json.loads(INT8.read_text())
    full = d["full_int8_drift_100_frames"]
    worst = max(max(v["int8_vs_fp_CPU"]["max"].values()) for v in full.values())
    conv = d["diag_pick_20_frames"]["int8_convonly"]
    conv_max = max(conv["max"].values())
    comp = {c["model"]: c for c in d["compress"] if c.get("variant", "").startswith("deployed")}
    c = comp.get("act-pick-v6-h100") or next(iter(comp.values()))
    txt = (f"full INT8 ({fmt(c['fp_bin_mb'], 0)} → {fmt(c['q_bin_mb'], 0)} MB) moves the joints by up to {fmt(worst, 0)}°, "
           f"unsafe; INT8 convolutions only ({fmt(conv['bin_mb'], 0)} MB) stay within {fmt(conv_max, 1)}°. The cell runs FP")
    return txt, rel(INT8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--since", default="20261114-1950")
    ap.add_argument("--truth", default="results/truth.csv")
    ap.add_argument("--smolvla", default=None, help="the SmolVLA sentence; default: still training at submission")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()

    groups = runs_by_inspector(a.since)
    if not a.no_report:
        run_report(None, CL, a.since, a.truth)
        for key, path in (("patchcore", CL_PC), ("classifier", CL_CLS)):
            if groups[key]:
                run_report(groups[key], path, a.since, a.truth)
    vals: dict[str, tuple[str, str]] = {}
    vals.update(group_vals("all", CL, a.truth))
    vals.update(group_vals("pc", CL_PC, a.truth))
    vals.update(group_vals("cls", CL_CLS, a.truth))
    th = cls_thresholds(groups["classifier"])
    if th:
        vals["cls_low"] = (fmt(th[0], 2), th[2])
        vals["cls_high"] = (fmt(th[1], 2), th[2])
    vals.update(classifier_photo_vals())
    ab = act_bench()
    for dev, key in (("GPU", "act_gpu"), ("NPU", "act_npu"), ("CPU", "act_cpu")):
        if dev in ab:
            v, src = ab[dev]
            vals[key] = (fmt(v, 1) if v < 50 else fmt(v, 0), src)
    vals["int8_line"] = int8_line()
    vals["smolvla_status"] = (a.smolvla or SMOLVLA_DEFAULT, "--smolvla" if a.smolvla else "default (SmolVLA v7 not tested)")

    text = Path(a.template).read_text(encoding="utf-8")
    for p in ("", "pc_", "cls_"):  # no hand labels for a group: drop its accuracy line instead of printing a dash
        if p + "labelled" not in vals:
            text = "\n".join(l for l in text.splitlines()
                             if not re.search(r"\{\{" + p + r"(accuracy|correct|labelled)\}\}", l)) + "\n"
    missing = []
    for k in sorted(set(re.findall(r"\{\{(\w+)\}\}", text))):
        v = vals.get(k)
        if v is None:
            missing.append(k)
            continue
        text = text.replace("{{" + k + "}}", v[0])
    Path(a.out).write_text(text, encoding="utf-8")

    used = set(re.findall(r"\{\{(\w+)\}\}", Path(a.template).read_text(encoding="utf-8")))
    lines = ["| placeholder | value | source |", "|---|---|---|"]
    for k, v in vals.items():
        if v is not None and k in used:
            lines.append(f"| {k} | {v[0][:90]} | {v[1]} |")
    for k in missing:
        lines.append(f"| {k} | MISSING | – |")
    lines += ["", "Runs per inspector: " + ", ".join(f"{k} {len(r)}" for k, r in groups.items())
              + f" (folders since {a.since} with a run.json)"]
    src = "\n".join(lines)
    Path(a.out + ".sources.md").write_text(src + "\n", encoding="utf-8")
    print(src)
    print(f"\nwrote {a.out} ({len(text)} chars, limit 5000)" + (f"; MISSING: {', '.join(missing)}" if missing else ""))
    return 1 if missing or len(text) > 5000 else 0


if __name__ == "__main__":
    sys.exit(main())
