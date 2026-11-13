"""Package power (RAPL) and temperature sampler for the Core Ultra laptop.

RAPL energy (package = CPU+iGPU+NPU, psys = whole laptop) is root-only on this kernel, so one `sudo -n` reader process streams it
(passwordless sudo is set up for the intel user). Temperatures come from /sys/class/thermal.

As a library:
    with PowerMonitor() as pm:
        run_workload()
    print(pm.summary())   # {'avg_w': ..., 'psys_avg_w': ..., 'energy_j': ..., 'x86_pkg_temp_max_c': ...}

As a CLI:
    python bench/power.py --idle 10                    # baseline
    python bench/power.py -- python my_inference.py    # wrap any command
"""
import argparse
import csv
import glob
import subprocess
import sys
import threading
import time
from pathlib import Path

DOMAINS = {"pkg": "/sys/class/powercap/intel-rapl:0", "psys": "/sys/class/powercap/intel-rapl:1"}  # SoC, whole platform
TEMP_ZONES = ("x86_pkg_temp", "TCPU")


def _zones():
    out = {}
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        name = Path(z, "type").read_text().strip()
        if name in TEMP_ZONES:
            out[name] = Path(z, "temp")
    return out


class PowerMonitor:
    def __init__(self, interval: float = 0.1, csv_path: str | None = None):
        self.interval = interval
        self.csv_path = csv_path
        self.samples = []  # (t, pkg_watts, psys_watts, {zone: celsius})
        self._zones = _zones()
        self._stop = threading.Event()

    def __enter__(self):
        files = " ".join(f"{d}/max_energy_range_uj" for d in DOMAINS.values())
        self._max = [int(v) for v in subprocess.run(["sudo", "-n", "sh", "-c", f"cat {files}"],
                                                    capture_output=True, text=True, check=True).stdout.split()]
        files = " ".join(f"{d}/energy_uj" for d in DOMAINS.values())
        self._proc = subprocess.Popen(
            ["sudo", "-n", "sh", "-c", f"while :; do echo $(cat {files}); sleep {self.interval}; done"],
            stdout=subprocess.PIPE, text=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        prev = None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            t, e = time.perf_counter(), [int(v) for v in line.split()]
            temps = {k: int(p.read_text()) / 1000 for k, p in self._zones.items()}
            if prev:
                dt = t - prev[0]
                w = [((b - a) % m) / 1e6 / dt for a, b, m in zip(prev[1], e, self._max)]  # % handles wraparound
                self.samples.append((t, w[0], w[1], temps))
            prev = (t, e)

    def __exit__(self, *exc):
        self._stop.set()
        self._proc.terminate()
        self._thread.join(timeout=1)
        if self.csv_path and self.samples:
            Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "pkg_w", "psys_w", *self._zones])
                t0 = self.samples[0][0]
                for t, pkg, psys, temps in self.samples:
                    w.writerow([round(t - t0, 3), round(pkg, 2), round(psys, 2), *temps.values()])

    def summary(self) -> dict:
        if not self.samples:
            return {}
        watts = [s[1] for s in self.samples]
        psys = [s[2] for s in self.samples]
        dur = self.samples[-1][0] - self.samples[0][0]
        out = {"seconds": round(dur, 2), "avg_w": round(sum(watts) / len(watts), 2),
               "max_w": round(max(watts), 2), "energy_j": round(sum(watts) / len(watts) * dur, 1),
               "psys_avg_w": round(sum(psys) / len(psys), 2)}
        for zone in self._zones:
            vals = [s[3][zone] for s in self.samples]
            out[f"{zone}_max_c"] = max(vals)
            out[f"{zone}_end_c"] = vals[-1]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", type=float, help="measure idle baseline for N seconds")
    ap.add_argument("--csv", help="write per-sample CSV here")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run while measuring")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd and not args.idle:
        ap.error("give --idle N or a command after --")
    with PowerMonitor(csv_path=args.csv) as pm:
        rc = subprocess.call(cmd) if cmd else time.sleep(args.idle)
    print(pm.summary())
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
