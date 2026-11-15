#!/usr/bin/env python3
"""Closed-loop numbers for the deck: what the cell did over real full cycles (pick -> inspect -> place).

Reads the cell program's run folders (data/cell_runs/<stamp>/run.json + cycles.jsonl, written by
tools/orchestrator.py or tools/cell_live.py) and the power log (results/power-*.csv from tools/power_log.py).
Every number in the output names the file it came from.

    python tools/closed_loop_report.py                               # every full-cycle run with inspection on
    python tools/closed_loop_report.py data/cell_runs/20261114-200200 data/cell_runs/20261114-195725
    python tools/closed_loop_report.py --since 20261114-1950 --truth results/truth.csv

--truth: a hand-filled CSV with header "run,cycle,truth" (truth = good | defect; cycle = the 0-based number in
cycles.jsonl). With it the report adds verdict accuracy and a confusion matrix (true class vs the bowl the
brick went to; an uncertain brick goes to the defect bowl, so it counts as "defect").
Runs with --no-inspection, --force-verdict or --only are skipped (they are not closed-loop runs).
Aborted cycles, half-written lines (power cut / kill) and runs that stopped early are counted, never dropped.
Output: results/closed_loop.md and results/closed_loop.json (--out to change).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import power_report as PR  # noqa: E402  (same folder: load_power, window_stats, quietest, newest_power_csv)

RUNS = ROOT / "data" / "cell_runs"
BOWL = {"place_good": "good", "place_bad": "defect"}


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 2)


def load_run(d: Path) -> tuple[dict, list[dict], int] | None:
    try:
        meta = json.loads((d / "run.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    recs, broken = [], 0
    p = d / "cycles.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                broken += 1  # a half-written line (power cut / kill)
    return meta, recs, broken


def closed_loop(meta: dict) -> tuple[bool, str]:
    a = meta.get("args") or {}
    if a.get("cmd") != "run":
        return False, "not a run"
    if a.get("only"):
        return False, f"--only {a['only']}"
    if a.get("no_inspection"):
        return False, "--no-inspection"
    if a.get("force_verdict"):
        return False, f"--force-verdict {a['force_verdict']}"
    return True, ""


def cycle_row(run: str, rec: dict) -> dict:
    stages = rec.get("stages") or []
    insp = rec.get("inspection") or {}
    place = [s for s in stages if s.get("name") in BOWL]
    picks = [s for s in stages if s.get("name") == "pick"]
    result = str(rec.get("result") or "unfinished")
    return {
        "run": run, "cycle": rec.get("cycle"), "started": rec.get("started"), "result": result,
        "sorted": result == "sorted", "cycle_s": rec.get("cycle_s"),
        "verdict": insp.get("verdict"), "score": insp.get("score"), "uncertain": insp.get("uncertain"),
        "low": insp.get("low"), "high": insp.get("high"), "model": insp.get("model"),
        "looks": insp.get("looks", 1 if insp else 0), "decided_by": insp.get("decided_by"),
        "infer_ms": insp.get("infer_ms"), "inspect_total_ms": insp.get("total_ms"),
        "bowl": BOWL.get(place[-1]["name"]) if place and rec.get("result") == "sorted" else None,
        "pick_attempts": len(picks), "place_attempts": len(place),
        "stages": [f"{s.get('name')}:{s.get('outcome')}:{s.get('run_s')}" for s in stages],
        "pick_s": next((s.get("run_s") for s in reversed(picks) if s.get("outcome") == "done"), None),
        "place_s": next((s.get("run_s") for s in reversed(place) if s.get("outcome") == "done"), None),
    }


def outcome_kind(result: str) -> str:
    r = result.lower()
    if r == "sorted":
        return "sorted"
    if "not at home" in r:
        return "aborted: arm not at HOME at cycle start"
    if "pick" in r:
        return "aborted: pick failed"
    if "spot" in r or "no brick" in r or "empty" in r:
        return "aborted: no brick on the spot"
    if "place" in r:
        return "aborted: place failed"
    if r.startswith("unfinished"):
        return "unfinished (stopped mid-cycle: killed / power cut)"
    return "aborted: " + result.split(";")[0][:60]


def fmt(v, unit: str = "") -> str:
    return "–" if v is None else f"{v}{unit}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path, help="run folders (default: all closed-loop runs)")
    ap.add_argument("--since", help="only runs whose folder name is >= this (e.g. 20261114-1950)")
    ap.add_argument("--truth", type=Path, default=ROOT / "results" / "truth.csv")
    ap.add_argument("--power", type=Path, help="power CSV (default: newest results/power-*.csv)")
    ap.add_argument("--idle-s", type=float, default=60.0, help="idle baseline = quietest N s of the power log")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "closed_loop")
    a = ap.parse_args()

    dirs = [d if d.is_absolute() else (Path.cwd() / d) for d in a.runs] or sorted(p for p in RUNS.iterdir() if p.is_dir())
    if a.since:
        dirs = [d for d in dirs if d.name >= a.since]
    rows, used, skipped = [], [], []
    for d in dirs:
        lr = load_run(d)
        if lr is None:
            skipped.append((d.name, "no run.json"))
            continue
        meta, recs, broken = lr
        ok, why = closed_loop(meta)
        if not ok:
            skipped.append((d.name, why))
            continue
        args = meta.get("args") or {}
        if not recs and not broken:
            skipped.append((d.name, "no cycle written"))
            continue
        try:
            src = str((d / "cycles.jsonl").relative_to(ROOT))
        except ValueError:
            src = str(d / "cycles.jsonl")
        used.append({"run": d.name, "device": args.get("device"), "anomaly_device": args.get("anomaly_device"),
                     "anomaly_model": args.get("anomaly_model"), "backend": args.get("backend"),
                     "execution": args.get("execution"), "pick_model": args.get("pick_model"),
                     "place_good_model": args.get("place_good_model"), "place_bad_model": args.get("place_bad_model"),
                     "reinspect_uncertain": args.get("reinspect_uncertain"), "cycles_planned": args.get("cycles"),
                     "cycles_written": len(recs), "broken_lines": broken, "source": src})
        rows += [cycle_row(d.name, r) for r in recs]
        for _ in range(broken):
            rows.append({"run": d.name, "cycle": None, "result": "unfinished (half-written line)", "sorted": False,
                         "cycle_s": None, "verdict": None, "bowl": None, "note": "unreadable line in cycles.jsonl"})
        planned = args.get("cycles")
        last_ok = bool(recs) and recs[-1].get("result") == "sorted"
        if isinstance(planned, int) and len(recs) + broken < planned and last_ok:
            # the run ended after a sorted cycle without writing the next one: killed or power cut mid-cycle
            rows.append({"run": d.name, "cycle": len(recs), "result": "unfinished", "sorted": False, "cycle_s": None,
                         "verdict": None, "bowl": None,
                         "note": f"{planned} cycles planned, {len(recs)} written, the last one sorted"})

    for r in rows:
        r["outcome"] = outcome_kind(r["result"])

    truth: dict[tuple[str, int], str] = {}
    if a.truth and a.truth.exists():
        with a.truth.open() as f:
            for t in csv.DictReader(f):
                try:
                    truth[(t["run"].strip(), int(t["cycle"]))] = t["truth"].strip().lower()
                except (KeyError, ValueError, AttributeError):
                    continue
    for r in rows:
        r["truth"] = truth.get((r["run"], r["cycle"])) if r.get("cycle") is not None else None

    sorted_rows = [r for r in rows if r["sorted"]]
    times = [r["cycle_s"] for r in sorted_rows if r.get("cycle_s")]
    outcomes: dict[str, int] = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

    power_csv = a.power or PR.newest_power_csv()
    energy = None
    if power_csv and power_csv.exists():
        prow = PR.load_power(power_csv)
        idle = PR.quietest(prow, a.idle_s) if prow else None
        idle_s = PR.window_stats(prow, *idle) if idle else None
        per = []
        for r in sorted_rows:
            if not r.get("started") or not r.get("cycle_s"):
                continue
            t0 = PR.parse_time(r["started"])
            s = PR.window_stats(prow, t0, t0 + r["cycle_s"])
            r["power"] = s
            if s:
                per.append(s)
        if per:
            psys = [s["psys_J"] for s in per if s.get("psys_J") is not None]
            pkg = [s["pkg_J"] for s in per if s.get("pkg_J") is not None]
            try:
                src = str(power_csv.relative_to(ROOT))
            except ValueError:
                src = str(power_csv)
            energy = {"source": src, "bricks_with_power": len(per),
                      "psys_J_per_brick_median": round(st.median(psys), 1) if psys else None,
                      "psys_Wh_per_brick_median": round(st.median(psys) / 3600, 4) if psys else None,
                      "pkg_J_per_brick_median": round(st.median(pkg), 1) if pkg else None,
                      "psys_W_mean_median": round(st.median([s["psys_W_mean"] for s in per if s.get("psys_W_mean")]), 1),
                      "pkg_C_max": max((s.get("pkg_C_max") or 0) for s in per),
                      "headroom_C_min": min((s.get("headroom_C_min") if s.get("headroom_C_min") is not None else 999)
                                            for s in per),
                      "npu_pct_mean_median": round(st.median([s.get("npu_pct_mean") or 0 for s in per]), 1),
                      "gpu_mhz_mean_median": round(st.median([s.get("gpu_mhz_mean") or 0 for s in per])),
                      "on_mains": all(s.get("on_mains") for s in per)}
            if idle_s and idle_s.get("psys_W_mean") is not None:
                iw = idle_s["psys_W_mean"]
                energy["idle"] = {"t0": idle[0], "t1": idle[1], "psys_W_mean": iw, "pkg_W_mean": idle_s.get("pkg_W_mean"),
                                  "how": f"quietest {a.idle_s:.0f} s window of the power log"}
                above = [s["psys_J"] - iw * s["seconds"] for s in per if s.get("psys_J") is not None]
                energy["psys_J_per_brick_above_idle_median"] = round(st.median(above), 1) if above else None

    lab = [r for r in sorted_rows if r.get("truth") in ("good", "defect") and r.get("bowl")]
    conf = {t: {b: sum(1 for r in lab if r["truth"] == t and r["bowl"] == b) for b in ("good", "defect")}
            for t in ("good", "defect")}
    acc = round(sum(1 for r in lab if r["truth"] == r["bowl"]) / len(lab), 3) if lab else None

    rep = {
        "made_by": "tools/closed_loop_report.py", "runs": used, "skipped_runs": skipped,
        "cycles": len(rows), "sorted": len(sorted_rows),
        "success_rate": round(len(sorted_rows) / len(rows), 3) if rows else None,
        "outcomes": outcomes,
        "cycle_s": {"n": len(times), "mean": round(st.mean(times), 2) if times else None,
                    "p50": pct(times, 0.5), "p95": pct(times, 0.95),
                    "min": min(times) if times else None, "max": max(times) if times else None},
        "pick_s_p50": pct([r["pick_s"] for r in sorted_rows if r.get("pick_s")], 0.5),
        "place_s_p50": pct([r["place_s"] for r in sorted_rows if r.get("place_s")], 0.5),
        "inspect_ms_p50": pct([r["inspect_total_ms"] for r in sorted_rows if r.get("inspect_total_ms")], 0.5),
        "infer_ms_per_frame_p50": pct([r["infer_ms"] for r in sorted_rows if r.get("infer_ms")], 0.5),
        "verdicts": {v: sum(1 for r in sorted_rows if r.get("verdict") == v) for v in ("good", "defect", "unsure")},
        "uncertain": sum(1 for r in sorted_rows if r.get("uncertain")),
        "reinspected": sum(1 for r in sorted_rows if (r.get("looks") or 0) >= 2),
        "bowls": {b: sum(1 for r in sorted_rows if r.get("bowl") == b) for b in ("good", "defect")},
        "truth_source": str(a.truth) if truth else None,
        "labelled": len(lab), "accuracy": acc, "confusion_truth_vs_bowl": conf if lab else None,
        "energy": energy, "rows": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=str))

    sr = rep["success_rate"]
    L = ["# Closed-loop results (pick -> inspect -> place, no human touch)", "",
         "Made by `tools/closed_loop_report.py` from the run folders below and the power log. "
         "Only runs with inspection on (no --no-inspection / --force-verdict / --only).", ""]
    L += ["## Runs", "", "| run | cycles written / planned | policy device | anomaly model / device | pick / good / bad model | source |",
          "|---|---|---|---|---|---|"]
    for u in used:
        L.append(f"| {u['run']} | {u['cycles_written']}/{u['cycles_planned']}"
                 f"{' (+' + str(u['broken_lines']) + ' broken)' if u['broken_lines'] else ''} "
                 f"| {u['device']} ({u['backend']}, {u['execution']}) | {u['anomaly_model']} / {u['anomaly_device']} "
                 f"| {u['pick_model']} / {u['place_good_model']} / {u['place_bad_model']} | `{u['source']}` |")
    if skipped:
        L += ["", f"Skipped {len(skipped)} run folders that are not closed-loop runs or wrote no cycle "
              f"(e.g. {', '.join(f'{n} ({w})' for n, w in skipped[-5:])})."]
    L += ["", "## Summary", "",
          f"- Cycles attempted: **{len(rows)}**, sorted without help: **{len(sorted_rows)}** "
          f"({fmt(round(100 * sr) if sr is not None else None, ' %')})",
          "- Outcomes: " + ", ".join(f"{k}: {v}" for k, v in sorted(outcomes.items(), key=lambda kv: -kv[1])),
          f"- Cycle time, sorted bricks (n={rep['cycle_s']['n']}): mean {fmt(rep['cycle_s']['mean'], ' s')}, "
          f"p50 {fmt(rep['cycle_s']['p50'], ' s')}, p95 {fmt(rep['cycle_s']['p95'], ' s')}; "
          f"pick p50 {fmt(rep['pick_s_p50'], ' s')}, place p50 {fmt(rep['place_s_p50'], ' s')}, "
          f"inspection p50 {fmt(rep['inspect_ms_p50'], ' ms')} (anomaly inference p50 {fmt(rep['infer_ms_per_frame_p50'], ' ms')} per frame of 3 views)",
          "- Verdicts on sorted bricks: " + ", ".join(f"{k} {v}" for k, v in rep["verdicts"].items() if v)
          + f"; uncertain {rep['uncertain']} (go to the defect bowl), re-inspected {rep['reinspected']}; "
          f"bowls: good {rep['bowls']['good']}, defect {rep['bowls']['defect']}"]
    if lab:
        L += [f"- Verdict accuracy vs hand labels ({len(lab)} bricks, `{a.truth}`): **{round(100 * acc)} %**", "",
              "| true \\ bowl | good bowl | defect bowl |", "|---|---|---|",
              f"| good | {conf['good']['good']} | {conf['good']['defect']} |",
              f"| defect | {conf['defect']['good']} | {conf['defect']['defect']} |", ""]
    else:
        L += ["- Verdict accuracy: no hand labels yet (fill `results/truth.csv` with run,cycle,truth)"]
    if energy:
        e = energy
        L += [f"- Energy per sorted brick (`{e['source']}`, {e['bricks_with_power']} bricks, median): whole laptop "
              f"{fmt(e['psys_J_per_brick_median'], ' J')} = {fmt(e['psys_Wh_per_brick_median'], ' Wh')}, CPU package "
              f"{fmt(e['pkg_J_per_brick_median'], ' J')}; mean draw {fmt(e['psys_W_mean_median'], ' W')}"
              + (f"; above idle ({e['idle']['psys_W_mean']} W, {e['idle']['how']}): "
                 f"{fmt(e.get('psys_J_per_brick_above_idle_median'), ' J')}" if e.get("idle") else "")
              + ("" if e.get("on_mains") else " (partly on battery)"),
              f"- Thermals during cycles: package max {fmt(e['pkg_C_max'], ' °C')}, min headroom to TjMax "
              f"{fmt(e['headroom_C_min'], ' °C')}; NPU busy {fmt(e['npu_pct_mean_median'], ' %')}, iGPU clock {fmt(e['gpu_mhz_mean_median'], ' MHz')} (medians of cycle means)"]
    else:
        L += ["- Energy: no power samples overlap these cycles"]
    L += ["", "## Cycles", "",
          "| run | # | outcome | s | verdict | score (low/high) | looks | bowl | truth | stages (name:outcome:run s) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rd = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "–"  # noqa: E731
        sc = (f"{r['score']:.2f} ({rd(r.get('low'))}/{rd(r.get('high'))})" if isinstance(r.get("score"), (int, float)) else "–")
        L.append(f"| {r['run']} | {fmt(r.get('cycle'))} | {r['outcome']} | {fmt(r.get('cycle_s'))} "
                 f"| {fmt(r.get('verdict'))}{' (uncertain)' if r.get('uncertain') else ''} | {sc} | {fmt(r.get('looks'))} "
                 f"| {fmt(r.get('bowl'))} | {fmt(r.get('truth'))} | {' '.join(r.get('stages') or []) or r.get('note', '')} |")
    a.out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {a.out.with_suffix('.md')} and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
