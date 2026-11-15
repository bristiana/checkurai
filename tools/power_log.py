#!/usr/bin/env python3
"""Log power, temperature and accelerator load on the Intel station laptop.

The RAPL energy counters are root-only, so run it with sudo (the output file is
handed back to the calling user):

    sudo -n "$(command -v python)" tools/power_log.py --out results/power-$(date +%m%d-%H%M).csv &

Stop it with:  sudo -n pkill -f tools/power_log.py   (or kill the pid in results/power_log.pid)

One CSV row every --interval seconds:
  t, iso         Unix time and local time (same clock as the orchestrator's cycles.jsonl)
  <dom>_J        energy since the logger started, wrap-corrected (J)
  <dom>_W        mean power over the last interval (W)
                 dom = pkg (CPU package incl. iGPU + NPU), core (CPU cores), uncore (RAPL PP1,
                 mostly the iGPU), dram, psys (whole platform, when the firmware exposes it)
  cpu_pct        all-core CPU busy % over the last interval
  pkg_C          CPU package temperature; headroom = tjmax_C - pkg_C
  hot_C          hottest of all CPU + Dell board sensors
  cpu_throttles  package thermal-throttle events since the logger started
  gpu_mhz        iGPU render clock (xe gt0 act_freq)
  gpu_throttle   iGPU throttle reasons active right now (pl1/pl2/pl4/prochot/ratl), "" when none
  npu_pct        NPU busy % over the last interval
  npu_mhz        NPU clock
  ac             1 on mains, 0 on battery
  bat_W          battery power (W): the whole laptop's draw when ac = 0
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import signal
import sys
import time
from pathlib import Path

RAPL_NAMES = {"package-0": "pkg", "core": "core", "uncore": "uncore", "dram": "dram", "psys": "psys"}


def read(path: str | Path, default: str | None = None) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def read_int(path: str | Path) -> int | None:
    v = read(path)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


class Rapl:
    """Cumulative, wrap-corrected energy per RAPL domain (only the MSR-based tree, not the MMIO duplicate)."""

    def __init__(self) -> None:
        self.doms: dict[str, tuple[Path, int]] = {}
        for z in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
            name = RAPL_NAMES.get(read(f"{z}/name", "") or "")
            if not name or name in self.doms or read_int(f"{z}/energy_uj") is None:
                continue
            self.doms[name] = (Path(z) / "energy_uj", read_int(f"{z}/max_energy_range_uj") or 2**32)
        self.last = {n: read_int(p) for n, (p, _) in self.doms.items()}
        self.total_uj = dict.fromkeys(self.doms, 0)

    def sample(self) -> dict[str, float]:
        out = {}
        for n, (p, rng) in self.doms.items():
            v = read_int(p)
            if v is None:
                continue
            d = v - self.last[n]
            if d < 0:
                d += rng
            self.last[n] = v
            self.total_uj[n] += d
            out[n] = d
        return out


class Temps:
    def __init__(self) -> None:
        self.pkg: Path | None = None
        self.tjmax: float | None = None
        self.all: list[Path] = []
        for h in glob.glob("/sys/class/hwmon/hwmon*"):
            name = read(f"{h}/name", "")
            if name not in ("coretemp", "dell_ddv", "dell_smm"):
                continue
            for t in glob.glob(f"{h}/temp*_input"):
                self.all.append(Path(t))
                if name == "coretemp" and read(t.replace("_input", "_label"), "") == "Package id 0":
                    self.pkg = Path(t)
                    crit = read_int(t.replace("_input", "_crit"))
                    self.tjmax = crit / 1000 if crit else None

    def sample(self) -> tuple[float | None, float | None]:
        vals = [v / 1000 for v in (read_int(p) for p in self.all) if v is not None and v > 0]
        pkg = read_int(self.pkg) if self.pkg else None
        return (pkg / 1000 if pkg is not None else None), (max(vals) if vals else None)


def cpu_times() -> tuple[int, int]:
    f = (read("/proc/stat", "") or "").splitlines()[0].split()[1:]
    v = [int(x) for x in f]
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return sum(v), idle


def throttle_count() -> int:
    return sum(read_int(p) or 0 for p in glob.glob("/sys/devices/system/cpu/cpu*/thermal_throttle/package_throttle_count")[:1])


def find_one(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="CSV path")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between rows (default 0.5)")
    ap.add_argument("--pidfile", default=None, help="write the pid here (default: <out dir>/power_log.pid)")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pidfile = Path(a.pidfile) if a.pidfile else out.parent / "power_log.pid"
    uid, gid = int(os.environ.get("SUDO_UID", os.getuid())), int(os.environ.get("SUDO_GID", os.getgid()))

    rapl, temps = Rapl(), Temps()
    if not rapl.doms:
        print("power_log: no readable RAPL counters (run it with sudo)", file=sys.stderr)
        return 1
    gpu_freq = find_one("/sys/class/drm/card*/device/tile0/gt0/freq0/act_freq")
    gpu_thr = sorted(glob.glob(os.path.join(os.path.dirname(gpu_freq), "throttle", "reason_*"))) if gpu_freq else []
    npu_dir = find_one("/sys/class/accel/accel*/device")
    ac_online = next((f"{d}/online" for d in glob.glob("/sys/class/power_supply/*") if read(f"{d}/type") == "Mains"), None)
    bat = next((d for d in glob.glob("/sys/class/power_supply/*") if read(f"{d}/type") == "Battery"), None)

    doms = list(rapl.doms)
    cols = ["t", "iso"] + [f"{d}_J" for d in doms] + [f"{d}_W" for d in doms] + [
        "cpu_pct", "pkg_C", "hot_C", "tjmax_C", "cpu_throttles", "gpu_mhz", "gpu_throttle", "npu_pct", "npu_mhz", "ac", "bat_W"]

    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    new = not out.exists() or out.stat().st_size == 0
    f = out.open("a", newline="", buffering=1)
    for p in (out, pidfile):
        if p == pidfile:
            p.write_text(f"{os.getpid()}\n")
        try:
            os.chown(p, uid, gid)
        except OSError:
            pass
    w = csv.writer(f)
    if new:
        w.writerow(cols)
    print(f"power_log: pid {os.getpid()}, {len(doms)} RAPL domains ({', '.join(doms)}), every {a.interval}s -> {out}", flush=True)

    t_prev = time.monotonic()
    cpu_prev = cpu_times()
    npu_prev = read_int(f"{npu_dir}/npu_busy_time_us") if npu_dir else None
    thr0 = throttle_count()
    next_t = t_prev + a.interval
    while not stop:
        time.sleep(max(0.0, next_t - time.monotonic()))
        next_t += a.interval
        now = time.monotonic()
        dt_s = max(now - t_prev, 1e-6)
        t_prev = now
        d_uj = rapl.sample()
        cpu = cpu_times()
        tot, idle = cpu[0] - cpu_prev[0], cpu[1] - cpu_prev[1]
        cpu_prev = cpu
        npu_now = read_int(f"{npu_dir}/npu_busy_time_us") if npu_dir else None
        npu_pct = None
        if npu_now is not None and npu_prev is not None:
            npu_pct = min(100.0, 100.0 * (npu_now - npu_prev) / (dt_s * 1e6))
        npu_prev = npu_now
        pkg_c, hot_c = temps.sample()
        reasons = [Path(p).name.removeprefix("reason_") for p in gpu_thr if read_int(p) == 1]
        bat_w = None
        if bat:
            pw = read_int(f"{bat}/power_now")
            if pw is None:  # some batteries expose current + voltage instead
                cur, volt = read_int(f"{bat}/current_now"), read_int(f"{bat}/voltage_now")
                pw = cur * volt / 1e6 if cur is not None and volt is not None else None
            bat_w = round(pw / 1e6, 2) if pw is not None else None
        ts = time.time()
        row = [round(ts, 3), dt.datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")]
        row += [round(rapl.total_uj[d] / 1e6, 4) for d in doms]
        row += [round(d_uj.get(d, 0) / 1e6 / dt_s, 3) for d in doms]
        row += [round(100.0 * (tot - idle) / tot, 1) if tot > 0 else "",
                pkg_c if pkg_c is not None else "", hot_c if hot_c is not None else "",
                temps.tjmax if temps.tjmax is not None else "", throttle_count() - thr0,
                read_int(gpu_freq) if gpu_freq else "", "+".join(reasons),
                round(npu_pct, 1) if npu_pct is not None else "",
                read_int(f"{npu_dir}/npu_current_frequency_mhz") if npu_dir else "",
                read(ac_online, "") if ac_online else "", bat_w if bat_w is not None else ""]
        w.writerow(row)
    f.close()
    try:
        if pidfile.read_text().strip() == str(os.getpid()):
            pidfile.unlink()
    except OSError:
        pass
    print("power_log: stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
