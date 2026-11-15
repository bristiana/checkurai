"""Merge Studio datasets (LeRobot v3) into one and import it into Studio as a new dataset.

For a language-conditioned policy (Pi0.5 / SmolVLA): place_good ("... lilac bowl") and place_bad ("... orange
bowl") become one dataset with two tasks, so one policy does both and the anomaly verdict picks the task string.

    python tools/dataset_merge.py --name place_both_v5 50677384-1d74-4183-aef4-235843eab71d 16871ef1-0d3d-4e62-bd42-e5d8de3c204b
    python tools/dataset_merge.py --name x --no-import <ids>     # merge only (to /tmp/dataset_merge/x)

Run with the intel_dev_env python (LeRobot 0.4.4). The source datasets are only read.
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
STUDIO_DATA = Path.home() / ".local/share/physicalai/datasets"
WORK = Path("/tmp/dataset_merge")


def api(cfg: dict, method: str, path: str, body: bytes | None = None, ctype: str | None = None) -> dict:
    req = urllib.request.Request(cfg.get("url", "http://127.0.0.1:7860") + path, data=body, method=method)
    if ctype:
        req.add_header("Content-Type", ctype)
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read() or b"{}")


def multipart(fields: dict[str, str], file_field: str | None = None, file_path: Path | None = None) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        out += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    if file_field:
        out += (f'--{b}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
                f"Content-Type: application/zip\r\n\r\n").encode()
        out += file_path.read_bytes() + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={b}"


def wait_job(cfg: dict, job_id: str, want: set[str], timeout: float = 1800) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = api(cfg, "GET", f"/api/jobs/{job_id}")
        step = (j.get("payload") or {}).get("step")
        if j.get("status") in ("failed", "canceled"):
            raise SystemExit(f"import job {j.get('status')}: {j.get('message')} {json.dumps(j.get('payload'))[:1500]}")
        if j.get("status") == "completed" or step in want:
            return j
        time.sleep(3)
    raise SystemExit("import job timed out")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="+", help="Studio dataset ids")
    ap.add_argument("--name", required=True, help="name of the merged dataset in Studio")
    ap.add_argument("--no-import", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    srcs = []
    for d in args.datasets:
        root = STUDIO_DATA / d
        ds = LeRobotDataset(f"local/{d[:8]}", root=root)
        print(f"{d[:8]}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames, tasks {list(ds.meta.tasks.index)}")
        srcs.append(ds)
    out = WORK / args.name
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = merge_datasets(srcs, output_repo_id=f"local/{args.name}", output_dir=out)
    print(f"merged -> {out}: {merged.meta.total_episodes} episodes, {merged.meta.total_frames} frames, "
          f"tasks {list(merged.meta.tasks.index)}")
    want_eps = sum(s.meta.total_episodes for s in srcs)
    if merged.meta.total_episodes != want_eps:
        raise SystemExit(f"episode count mismatch: {merged.meta.total_episodes} != {want_eps}")
    if args.no_import:
        return 0

    zpath = WORK / f"{args.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out).as_posix())
    print(f"archive {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")

    cfg = json.loads((ROOT / "config" / "cell.json").read_text())
    p = cfg["project"]
    envs = api(cfg, "GET", f"/api/projects/{p}/environments")
    env = next(e for e in envs if cfg.get("environment", "pick") in (e["id"], e["name"]))
    body, ct = multipart({"format_hint": "auto", "dataset_name": args.name})
    job = api(cfg, "POST", f"/api/projects/{p}/imports/datasets:prepare", body, ct)
    jid = job["id"]
    body, ct = multipart({}, "archive", zpath)
    api(cfg, "PUT", f"/api/projects/{p}/imports/datasets/{jid}:upload", body, ct)
    j = wait_job(cfg, jid, {"awaiting_user_review", "awaiting_finalize", "ready_to_finalize", "validated"})
    rep = (j.get("payload") or {}).get("validation_report")
    print("validated:", (j.get("payload") or {}).get("step"), json.dumps(rep)[:600] if rep else "")
    if j.get("status") != "completed":
        fin = json.dumps({"environment_id": env["id"], "default_task": ""}).encode()
        api(cfg, "POST", f"/api/projects/{p}/imports/datasets/{jid}:finalize", fin, "application/json")
        j = wait_job(cfg, jid, set())
    did = (j.get("payload") or {}).get("result_dataset_id")
    print(f"imported into Studio as {args.name!r}: dataset id {did}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
