#!/usr/bin/env python3
"""Turn a power_log.py CSV into numbers for the deck: per cycle, per window, or since a time.

    python tools/power_report.py --cycles data/cell_runs/<stamp>/cycles.jsonl      # per sorting cycle
    python tools/power_report.py --from 2026-11-14T14:00 --to 2026-11-14T14:20     # any window
    python tools/power_report.py --last 10                                         # last 10 minutes

The power CSV defaults to the newest results/power-*.csv. --idle START END (or --idle-s N =
the N quietest seconds of the log) gives a baseline, so the report can also say how much
energy the cell itself added on top of an idle laptop. Add --json PATH to save the numbers.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOMS = ("pkg", "core", "uncore", "dram", "psys")


def newest_power_csv() -> Path | None:
    hits = sorted((ROOT / "results").glob("power-*.csv"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def load_power(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("t") == "t":  # a second header after an appended restart
                continue
            out = {}
            for k, v in r.items():
                if k in ("iso", "gpu_throttle"):
                    out[k] = v or ""
                else:
                    try:
                        out[k] = float(v) if v not in ("", None) else None
                    except ValueError:
                        out[k] = None
            rows.append(out)
    # A logger restart resets the cumulative *_J columns; re-base them so they keep rising.
    off = dict.fromkeys(DOMS, 0.0)
    prev: dict | None = None
    for r in rows:
        for d in DOMS:
            k = f"{d}_J"
            if r.get(k) is None:
                continue
            if prev is not None and prev.get(k) is not None and r[k] + off[d] < prev[k]:
                off[d] = prev[k]
            r[k] += off[d]
        prev = r
    return rows


def parse_time(s: str) -> float:
    return dt.datetime.fromisoformat(s).timestamp()


def window_stats(rows: list[dict], t0: float, t1: float) -> dict | None:
    """Energy (exact, from the cumulative counters) and mean/peak power, temperatures and loads in [t0, t1]."""
    w = [r for r in rows if t0 <= r["t"] <= t1]
    if len(w) < 2:
        return None
    dur = w[-1]["t"] - w[0]["t"]
    out: dict = {"seconds": round(dur, 2), "samples": len(w)}
    for d in DOMS:
        a, b = w[0].get(f"{d}_J"), w[-1].get(f"{d}_J")
        if a is None or b is None:
            continue
        out[f"{d}_J"] = round(b - a, 2)
        out[f"{d}_W_mean"] = round((b - a) / dur, 2) if dur > 0 else None
        ws = [r[f"{d}_W"] for r in w[1:] if r.get(f"{d}_W") is not None]
        if ws:
            out[f"{d}_W_peak"] = round(max(ws), 2)

    def col(k: str) -> list[float]:
        return [r[k] for r in w if r.get(k) is not None]

    for k, agg in (("pkg_C", max), ("hot_C", max), ("cpu_pct", st.mean), ("gpu_mhz", st.mean),
                   ("npu_pct", st.mean), ("bat_W", st.mean)):
        v = col(k)
        if v:
            out[f"{k}_{'max' if agg is max else 'mean'}"] = round(agg(v), 1)
    tj = col("tjmax_C")
    if tj and out.get("pkg_C_max") is not None:
        out["headroom_C_min"] = round(tj[0] - out["pkg_C_max"], 1)
    thr = col("cpu_throttles")
    out["cpu_throttle_events"] = int(thr[-1] - thr[0]) if thr else None
    reasons = sorted({x for r in w for x in (r.get("gpu_throttle") or "").split("+") if x})
    out["gpu_throttle_reasons"] = reasons
    out["gpu_throttle_pct"] = round(100 * sum(1 for r in w if r.get("gpu_throttle")) / len(w), 1)
    ac = col("ac")
    out["on_mains"] = bool(ac and min(ac) == 1)
    return out


def quietest(rows: list[dict], seconds: float) -> tuple[float, float] | None:
    """The window of the given length with the lowest package energy (an idle baseline)."""
    best = None
    j = 0
    for i, r in enumerate(rows):
        while j < len(rows) and rows[j]["t"] - r["t"] < seconds:
            j += 1
        if j >= len(rows):
            break
        e = rows[j]["pkg_J"] - r["pkg_J"]
        if best is None or e < best[0]:
            best = (e, r["t"], rows[j]["t"])
    return (best[1], best[2]) if best else None


def cycle_windows(path: Path) -> list[tuple[str, float, float, dict]]:
    out = []
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        start = parse_time(rec["started"])
        if "cycle_s" in rec:  # full cycle: "started" is written when the cycle begins
            t0, t1 = start, start + rec["cycle_s"]
        else:  # single-stage run (--only): "started" is written after the stage finished
            busy = sum((s.get("load_s") or 0) + (s.get("run_s") or 0) for s in rec.get("stages", []))
            t0, t1 = start - busy, start
        label = rec.get("only") or f"cycle {rec.get('cycle')}"
        out.append((label, t0, t1, rec))
    return out


def fmt(v, unit: str = "", nd: int = 1) -> str:
    return "–" if v is None else f"{v:.{nd}f}{unit}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--power", type=Path, default=None, help="power CSV (default: newest results/power-*.csv)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cycles", type=Path, help="orchestrator cycles.jsonl")
    g.add_argument("--from", dest="t_from", help="window start, local ISO time (use with --to)")
    g.add_argument("--last", type=float, help="the last N minutes")
    ap.add_argument("--to", dest="t_to", help="window end, local ISO time (default: now)")
    ap.add_argument("--idle", nargs=2, metavar=("START", "END"), help="idle baseline window, local ISO times")
    ap.add_argument("--idle-s", type=float, default=None, help="use the quietest N seconds of the log as the idle baseline")
    ap.add_argument("--json", type=Path, help="also save the numbers here")
    a = ap.parse_args()

    pcsv = a.power or newest_power_csv()
    if not pcsv or not pcsv.exists():
        print("no power CSV (start tools/power_log.py first)", file=sys.stderr)
        return 1
    rows = load_power(pcsv)
    if len(rows) < 2:
        print(f"{pcsv}: not enough samples yet", file=sys.stderr)
        return 1

    idle = None
    if a.idle:
        idle = window_stats(rows, parse_time(a.idle[0]), parse_time(a.idle[1]))
    elif a.idle_s:
        q = quietest(rows, a.idle_s)
        idle = window_stats(rows, *q) if q else None
    idle_w = idle.get("pkg_W_mean") if idle else None

    report: dict = {"power_csv": str(pcsv.relative_to(ROOT) if pcsv.is_relative_to(ROOT) else pcsv), "idle": idle}
    print(f"power log: {report['power_csv']}  ({len(rows)} samples)")
    if idle:
        print(f"idle baseline: package {fmt(idle.get('pkg_W_mean'), ' W')}, platform {fmt(idle.get('psys_W_mean'), ' W')} "
              f"over {idle['seconds']:.0f} s")

    if a.cycles:
        wins = cycle_windows(a.cycles)
        report["cycles"] = []
        print(f"\n| run | result | s | pkg J | pkg W mean / peak | platform J | iGPU W | max °C (headroom) | NPU % |")
        print("|---|---|---|---|---|---|---|---|---|")
        for label, t0, t1, rec in wins:
            s = window_stats(rows, t0, t1)
            result = rec.get("result") or "/".join(f"{x['name']}:{x['outcome']}" for x in rec.get("stages", []))
            report["cycles"].append({"label": label, "result": result, "t0": t0, "t1": t1, "power": s})
            if not s:
                print(f"| {label} | {result} | {t1 - t0:.1f} | (no power samples) |||||| ")
                continue
            print(f"| {label} | {result} | {s['seconds']:.1f} | {fmt(s.get('pkg_J'))} | {fmt(s.get('pkg_W_mean'))} / {fmt(s.get('pkg_W_peak'))} "
                  f"| {fmt(s.get('psys_J'))} | {fmt(s.get('uncore_W_mean'))} | {fmt(s.get('pkg_C_max'), '', 0)} ({fmt(s.get('headroom_C_min'), '', 0)}) "
                  f"| {fmt(s.get('npu_pct_mean'))} |")
        good = [c for c in report["cycles"] if c["power"] and c["result"] == "sorted"]
        if good:
            e = [c["power"]["pkg_J"] for c in good]
            sec = [c["power"]["seconds"] for c in good]
            summ = {"sorted": len(good), "pkg_J_per_brick_median": round(st.median(e), 1),
                    "cycle_s_median": round(st.median(sec), 1)}
            if idle_w is not None:
                summ["pkg_J_per_brick_above_idle_median"] = round(st.median(x - idle_w * s for x, s in zip(e, sec)), 1)
            report["summary"] = summ
            print(f"\nsorted bricks: {len(good)}, median {summ['cycle_s_median']} s and {summ['pkg_J_per_brick_median']} J (package) per brick"
                  + (f", {summ['pkg_J_per_brick_above_idle_median']} J above idle" if "pkg_J_per_brick_above_idle_median" in summ else ""))
    else:
        if a.last:
            t1 = rows[-1]["t"]
            t0 = t1 - a.last * 60
        else:
            t0 = parse_time(a.t_from)
            t1 = parse_time(a.t_to) if a.t_to else rows[-1]["t"]
        s = window_stats(rows, t0, t1)
        report["window"] = {"t0": t0, "t1": t1, "power": s}
        if not s:
            print("no samples in that window")
        else:
            print(json.dumps(s, indent=1))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=1))
        print(f"saved {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
