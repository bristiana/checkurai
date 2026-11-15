#!/usr/bin/env python3
"""Cell program: pick a brick onto the inspection spot, inspect it, sort it into a bowl.

One cycle:
  1. spot check  the inspection camera must see an empty spot (a brick already there is inspected directly)
  2. pick        ACT policy "pick" runs until the arm is back at HOME and still
  3. spot check  a brick must now be on the spot, otherwise the pick is retried
  4. inspect     N frames from the inspection camera, median anomaly score: good / defect / uncertain
  5. place       ACT policy "place_good" or "place_bad"; an uncertain brick is inspected again (fresh frames,
                 same pose); still uncertain -> the defective bowl
  6. spot check  the spot must be empty again
  7. log         one JSON line per cycle, plus the inspection crop and a top-camera frame

The arm is only driven through Physical AI Studio's runtime session websocket
(/api/projects/<project>/runtime/ws): load_environment, load_model, then set_control with a
model control. Between stages the follower is held where it is (set_control null), so a
stage that times out or stalls leaves the arm held, not limp. The session holds the robot
while this program runs, so close Studio's record / inference page first.

    # learn HOME from the place_good demos (start and end of every episode)
    python tools/orchestrator.py learn-home --dataset <place_good dataset id>
    # check the done detector on recorded demos (no robot needed)
    python tools/orchestrator.py replay --dataset <dataset id>
    # connect and report; never loads anything and never moves the arm
    python tools/orchestrator.py dry-run
    # one stage at a time for the first live tests, then the loop
    python tools/orchestrator.py run --only pick
    python tools/orchestrator.py run --cycles 20

While a stage runs, type s + Enter to hold the arm and stop. Run it with the Python that has
Anomalib / OpenVINO / aiohttp (the intel_dev_env conda env on the station laptop).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STUDIO_DATA = Path.home() / ".local" / "share" / "physicalai"
DEFAULT_URL = "http://127.0.0.1:7860"
DEFAULT_HOME = ROOT / "config" / "home_pose.json"
DEFAULT_CELL_CONFIG = ROOT / "config" / "cell.json"
RUNS_DIR = ROOT / "data" / "cell_runs"  # data/ is gitignored
STUDIO_PY = Path.home() / "physical-ai-studio" / "application" / "backend" / ".venv" / "bin" / "python"  # has OpenVINO

TASKS = {
    "pick": "pick up the brick and put it on the spot",
    "place_good": "put the brick from the spot into the lilac bowl",
    "place_bad": "put the brick from the spot into the orange bowl",
}


def joint_key(name: str) -> str:
    """Studio and the datasets name joints 'shoulder_pan.pos'; compare them without the suffix."""
    return name.removesuffix(".pos")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


# --------------------------------------------------------------------------------------------
# HOME pose and the "stage is done" detector
# --------------------------------------------------------------------------------------------


@dataclass
class HomePose:
    """Where every demo starts and ends, with a per-joint tolerance."""

    names: list[str]
    q: np.ndarray
    tol: np.ndarray
    source: dict[str, Any] = field(default_factory=dict)

    def deviation(self, q: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(q, dtype=np.float64) - self.q)

    def score(self, q: np.ndarray) -> float:
        """max over joints of |q - home| / tolerance; <= 1 means at HOME."""
        return float(np.max(self.deviation(q) / self.tol))

    def near(self, q: np.ndarray) -> bool:
        return self.score(q) <= 1.0

    def to_json(self) -> dict[str, Any]:
        return {
            "joints": self.names,
            "home": [round(float(v), 3) for v in self.q],
            "tolerance": [round(float(v), 3) for v in self.tol],
            "source": self.source,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> HomePose:
        return cls([joint_key(n) for n in d["joints"]], np.asarray(d["home"], float), np.asarray(d["tolerance"], float),
                   d.get("source", {}))

    @classmethod
    def load(cls, path: Path) -> HomePose:
        return cls.from_json(json.loads(Path(path).read_text()))


def dataset_dir(dataset: str) -> Path:
    p = Path(dataset)
    return p if p.is_dir() else STUDIO_DATA / "datasets" / dataset


def load_episodes(dataset: str) -> tuple[list[str], float, dict[int, np.ndarray]]:
    """(joint names, fps, {episode index: state array [T, J]}) from a LeRobot-format dataset."""
    import pandas as pd

    root = dataset_dir(dataset)
    info = json.loads((root / "meta" / "info.json").read_text())
    names = [joint_key(n) for n in info["features"]["observation.state"]["names"]]
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"no episodes in {root}")
    df = pd.concat([pd.read_parquet(f, columns=["observation.state", "episode_index", "frame_index"]) for f in files])
    df = df.sort_values(["episode_index", "frame_index"])
    episodes = {int(e): np.stack(g["observation.state"].to_numpy()).astype(np.float64)
                for e, g in df.groupby("episode_index")}
    return names, float(info["fps"]), episodes


def learn_home(dataset: str, edge_frames: int = 15, scale: float = 2.0, floor: float = 6.0, cap: float = 20.0,
               gripper_cap: float = 30.0) -> HomePose:
    """HOME = mean of the first and last `edge_frames` of every episode; tolerance from their spread."""
    names, fps, episodes = load_episodes(dataset)
    edges = []
    for s in episodes.values():
        edges.append(s[:edge_frames].mean(0))
        edges.append(s[-edge_frames:].mean(0))
    edges = np.asarray(edges)
    home = edges.mean(0)
    spread = np.percentile(np.abs(edges - home), 95, axis=0)
    caps = np.array([gripper_cap if n == "gripper" else cap for n in names])
    tol = np.clip(scale * spread, floor, caps)
    return HomePose(names, home, tol, {
        "dataset": str(dataset), "episodes": len(episodes), "fps": fps, "edge_frames": edge_frames,
        "spread_p95": [round(float(v), 3) for v in spread], "learned_at": now_iso(),
    })


class DoneDetector:
    """A stage is done when the arm has left HOME, come back to it, and stayed still.

    update() returns one of:
      "start"    has not left HOME yet
      "moving"   away from HOME (or still settling)
      "done"     back at HOME and still for `still_s`, or within `hold_score` x tolerance of HOME for `hold_s`
                 (on the arm, Sun 03:00, a finished pick sat at HOME for 18 s without counting as done: the policy
                 keeps twitching a joint by more than still_range there, or rests just outside the tolerance)
      "stalled"  still for `stall_s` but NOT at HOME (the policy is stuck, e.g. hovering)
    """

    def __init__(self, home: HomePose, still_s: float = 0.7, still_range: float = 1.5, gripper_still_range: float = 3.0,
                 leave_score: float = 2.0, stall_s: float = 4.0, hold_s: float = 1.5, hold_score: float = 1.4):
        self.home = home
        self.still_s = still_s
        self.stall_s = stall_s
        self.hold_s = hold_s
        self.hold_score = hold_score
        self.leave_score = leave_score
        self.still_range = np.array([gripper_still_range if n == "gripper" else still_range for n in home.names])
        self.reset()

    def reset(self) -> None:
        self.buf: list[tuple[float, np.ndarray]] = []
        self.left = False
        self.t_left: float | None = None
        self.still_since: float | None = None
        self.home_since: float | None = None

    def _still(self, t: float) -> bool:
        """Has every joint stayed within its range over the last still_s seconds?"""
        if not self.buf or t - self.buf[0][0] < self.still_s * 0.95:
            return False
        q = np.stack([b[1] for b in self.buf])
        return bool(np.all(q.max(0) - q.min(0) <= self.still_range))

    def update(self, t: float, q: np.ndarray) -> str:
        q = np.asarray(q, dtype=np.float64)
        self.buf.append((t, q))
        while self.buf and t - self.buf[0][0] > self.still_s:
            self.buf.pop(0)
        if not self.left and self.home.score(q) > self.leave_score:
            self.left, self.t_left = True, t
        still = self._still(t)
        if still:
            self.still_since = self.still_since if self.still_since is not None else t - self.still_s
        else:
            self.still_since = None
        if not self.left:
            return "start"
        if self.home.score(q) <= self.hold_score:
            self.home_since = self.home_since if self.home_since is not None else t
        else:
            self.home_since = None
        if still and self.home.near(q):
            return "done"
        if self.home_since is not None and t - self.home_since >= self.hold_s:
            return "done"
        if still and self.still_since is not None and t - self.still_since >= self.stall_s:
            return "stalled"
        return "moving"


def replay(dataset: str, home: HomePose, det_kw: dict[str, Any]) -> dict[str, Any]:
    """Run the detector over every recorded episode; it should fire once, in the final hold."""
    names, fps, episodes = load_episodes(dataset)
    order = [names.index(n) for n in home.names]
    rows = []
    for e, s in episodes.items():
        det = DoneDetector(home, **det_kw)
        fired, first_left, stalled = None, None, None
        for i, q in enumerate(s[:, order]):
            st = det.update(i / fps, q)
            if det.left and first_left is None:
                first_left = i / fps
            if st == "stalled" and stalled is None:
                stalled = i / fps
            if st == "done":
                fired = i / fps
                break
        dur = (len(s) - 1) / fps
        rows.append({"episode": e, "duration_s": round(dur, 2), "left_s": first_left, "done_s": fired,
                     "before_end_s": None if fired is None else round(dur - fired, 2),
                     "stalled_s": stalled, "start_score": round(home.score(s[0, order]), 2),
                     "end_score": round(home.score(s[-1, order]), 2)})
    fired = [r for r in rows if r["done_s"] is not None]
    return {
        "dataset": str(dataset), "episodes": len(rows), "fired": len(fired),
        "fired_in_last_2s": sum(1 for r in fired if r["before_end_s"] <= 2.0),
        "median_before_end_s": float(np.median([r["before_end_s"] for r in fired])) if fired else None,
        "rows": rows,
    }


# --------------------------------------------------------------------------------------------
# Studio runtime session (websocket)
# --------------------------------------------------------------------------------------------


class StudioError(RuntimeError):
    pass


class StudioSession:
    """One runtime session in Studio, driven over its websocket.

    Commands are acked in order. The server pushes `state` (what is loaded, what drives the
    follower), `status` per part, `error`, and a binary observation per tick:
    [u32 big-endian header length][header JSON][JPEG bytes in header.frames order].
    """

    def __init__(self, url: str, project_id: str, hz: float = 30.0):
        self.url = url.rstrip("/")
        self.project_id = project_id
        self.hz = hz
        self.state: dict[str, Any] = {}
        self.statuses: dict[str, dict[str, Any]] = {}
        self.errors: list[dict[str, Any]] = []
        self.joints: dict[str, float] = {}
        self.joints_t = 0.0
        self.frames: dict[str, tuple[float, bytes]] = {}
        self._pending: dict[str, asyncio.Future] = {}
        self._got_state = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aenter__(self) -> StudioSession:
        import aiohttp

        self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10))
        ws_url = self.url.replace("http", "ws", 1) + f"/api/projects/{self.project_id}/runtime/ws?hz={self.hz:g}"
        self._ws = await self._http.ws_connect(ws_url, max_msg_size=0, heartbeat=20)
        self._reader = asyncio.create_task(self._read())
        await asyncio.wait_for(self._got_state.wait(), 10)
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._reader.cancel()
        await self._ws.close()
        await self._http.close()

    # ---- REST helpers
    async def get(self, path: str) -> Any:
        async with self._http.get(self.url + path) as r:
            if r.status >= 400:
                raise StudioError(f"GET {path}: {r.status} {await r.text()}")
            return await r.json()

    # ---- websocket
    async def _read(self) -> None:
        import aiohttp

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._on_json(json.loads(msg.data))
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    self._on_observation(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        finally:
            self.closed.set()
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(StudioError("runtime session websocket closed"))

    def _on_json(self, m: dict[str, Any]) -> None:
        kind = m.get("type")
        if kind == "ack":
            fut = self._pending.pop(m.get("request_id", ""), None)
            if fut is not None and not fut.done():
                fut.set_result(m)
        elif kind == "state":
            self.state = m["data"]
            self._got_state.set()
        elif kind == "status":
            self.statuses[m["component"]] = m
        elif kind == "error":
            self.errors.append(m)
            print(f"[studio] error from {m.get('component')}: {m.get('message')} ({m.get('error_code')})", flush=True)

    def _on_observation(self, data: bytes) -> None:
        n = int.from_bytes(data[:4], "big")
        header = json.loads(data[4:4 + n])
        off = 4 + n
        for f in header.get("frames", []):
            self.frames[f["key"]] = (f["timestamp"], bytes(data[off:off + f["size"]]))
            off += f["size"]
        if header.get("state"):
            self.joints = {joint_key(k): v for k, v in header["state"].items()}
            self.joints_t = time.monotonic()

    async def command(self, name: str, timeout: float = 120.0, **fields: Any) -> dict[str, Any] | None:
        if self.closed.is_set():
            raise StudioError("runtime session websocket is closed")
        rid = uuid.uuid4().hex
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._ws.send_json({"command": name, "request_id": rid, **fields})
        ack = await asyncio.wait_for(fut, timeout)
        if not ack.get("ok"):
            err = ack.get("error") or {}
            raise StudioError(f"{name} failed: {err.get('message')} ({err.get('error_code')})")
        return ack.get("result")

    def failed_parts(self) -> list[str]:
        return [f"{c}: {s.get('error')}" for c, s in self.statuses.items() if s.get("state") == "failed"]

    def q(self, names: list[str]) -> np.ndarray | None:
        if not self.joints:
            return None
        return np.array([self.joints[n] for n in names], dtype=np.float64)

    # ---- the parts
    async def load_environment(self, env: str) -> dict[str, Any]:
        envs = await self.get(f"/api/projects/{self.project_id}/environments")
        saved = next((e for e in envs if env in (e["id"], e["name"])), None)
        if saved is None:
            raise StudioError(f"no environment {env!r}; have {[e['name'] for e in envs]}")
        draft = {"id": saved["id"], "name": saved["name"], "robots": saved.get("robots", []),
                 "cameras": saved.get("cameras", [])}
        await self.command("load_environment", environment=draft, unload_incompatible=False, timeout=180)
        return saved

    async def resolve_model(self, model: str) -> dict[str, Any]:
        models = await self.get(f"/api/projects/{self.project_id}/models")
        hits = [m for m in models if model in (m["id"], m["name"])]
        if not hits:
            raise StudioError(f"no model {model!r}; have {[m['name'] for m in models]}")
        return max(hits, key=lambda m: m.get("version") or 0)

    def loaded_model_id(self) -> str | None:
        m = self.state.get("model")
        return m["model_id"] if m else None

    async def load_model(self, model_id: str, backend: str, device: str, force: bool = False) -> float:
        """Load (or keep) a model; returns the seconds the swap took (0 if it was already loaded). force: load it
        again even if it is loaded (its export changed; Studio unloads the loaded one first)."""
        if not force and self.loaded_model_id() == model_id:
            return 0.0
        t = time.perf_counter()
        await self.command("load_model", model_id=model_id, backend=backend, device=device, timeout=300)
        return time.perf_counter() - t

    async def run_model(self, task: str, hz: float, mode: str, threshold: float, grace_s: float) -> None:
        control = {"type": "model", "task": task, "hz": hz,
                   "execution": {"mode": mode, "request_threshold": threshold}, "grace_s": grace_s}
        await self.command("set_control", control=control, timeout=60)

    async def teleop(self, grace_s: float = 3.0) -> None:
        await self.command("set_control", control={"type": "teleop", "grace_s": grace_s}, timeout=60)

    async def hold(self) -> None:
        """Stop whatever drives the follower; it stays where it is."""
        await self.command("set_control", control=None, timeout=60)


async def list_sessions(url: str) -> list[dict[str, Any]]:
    """The live runtime sessions (Studio sends the list once on connect)."""
    import aiohttp

    ws_url = url.rstrip("/").replace("http", "ws", 1) + "/api/runtime/sessions/ws"
    async with aiohttp.ClientSession() as http, http.ws_connect(ws_url) as ws:
        msg = await ws.receive_json(timeout=5)
        return msg.get("sessions", [])


# --------------------------------------------------------------------------------------------
# Inspection camera and inspector
# --------------------------------------------------------------------------------------------


class InspectionCamera:
    """The inspection webcam at full resolution (MJPG), read continuously in a thread."""

    def __init__(self, name: str = "Webcam Audio", device: str | None = None, width: int = 1920, height: int = 1080,
                 fps: int = 30):
        self.name, self.device, self.width, self.height, self.fps = name, device, width, height, fps
        self._latest: np.ndarray | None = None
        self._seq = 0
        self.aligner = None  # tools.frame_align.FrameAligner, set by cmd_run
        self._cond = threading.Condition()
        self._stop = threading.Event()

    def start(self) -> InspectionCamera:
        import cv2

        from tools.capture_inspection import find_camera_nodes

        nodes = [self.device] if self.device else find_camera_nodes(self.name)
        for dev in nodes:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = False, None
            for _ in range(10):  # let auto exposure settle
                ok, frame = cap.read()
            if ok and frame is not None and frame.shape[1] == self.width and frame.shape[0] == self.height:
                self.cap, self.device = cap, dev
                self._thread = threading.Thread(target=self._loop, name="inspection-camera", daemon=True)
                self._thread.start()
                return self
            cap.release()
        raise RuntimeError(f"cannot read {self.width}x{self.height} frames from {nodes or self.name!r}. Is the camera "
                           "busy? Close Studio's preview of it and check with: fuser /dev/videoN")

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                with self._cond:
                    self._latest, self._seq = frame, self._seq + 1
                    self._cond.notify_all()

    def grab(self, n: int, timeout: float = 5.0) -> list[np.ndarray]:
        """The next n frames captured after this call."""
        out: list[np.ndarray] = []
        deadline = time.monotonic() + timeout
        with self._cond:
            seq = self._seq
            while len(out) < n:
                newer = lambda last=seq: self._seq > last  # noqa: E731
                if not self._cond.wait_for(newer, timeout=max(0.0, deadline - time.monotonic())):
                    raise TimeoutError(f"inspection camera: got {len(out)} of {n} frames")
                seq = self._seq
                out.append(self._latest.copy())
        if self.aligner is not None:  # undo a bumped camera (tools/frame_align.py)
            out = [self.aligner.align(f) for f in out]
        return out

    def stop(self) -> None:
        self._stop.set()
        if hasattr(self, "cap"):
            self._thread.join(timeout=2)
            self.cap.release()


@dataclass
class Inspection:
    verdict: str  # "good" | "defect" | "empty"
    score: float | None
    uncertain: bool
    brick_fraction: float
    frames: int
    infer_ms: float
    low: float | None = None
    high: float | None = None
    model: str = ""
    crop: np.ndarray | None = field(default=None, repr=False)
    frame_scores: list[float] = field(default_factory=list)  # per frame, to see how much a re-inspection can move
    frame: np.ndarray | None = field(default=None, repr=False)  # the median-score full frame (evidence, re-scoring)

    def log(self) -> dict[str, Any]:
        d = asdict(replace(self, crop=None, frame=None))
        d.pop("crop")
        d.pop("frame")
        return d


class Inspector(Protocol):
    def spot(self, frames: list[np.ndarray]) -> tuple[bool, float]:
        """(is a brick on the spot, changed fraction of the ROI)."""

    def inspect(self, frames: list[np.ndarray]) -> Inspection: ...


class AnomalibInspector:
    """ROI crop, empty-spot check and the exported Anomalib model (tools/inspection.py).

    Decision on the median score of the frames: < low good; >= high defect; in between
    defect with uncertain=True (an uncertain brick goes to the defective bowl).
    """

    def __init__(self, model_name: str, device: str = "GPU", force: str | None = None):
        from tools import inspection as insp

        self.insp = insp
        mdir = ROOT / "models" / "anomaly"
        res = json.loads((ROOT / "results" / "anomaly" / f"{model_name}.json").read_text())
        # eval-views --mask results (<name>_mask): the same exported model ("base_model"), each view scored only
        # inside the brick outline ("mask"), so paper, tape and mirror edges cannot make a clean brick a defect;
        # "exposure": each view normalised to its empty reference's exposure first (eval-views --exposure)
        base = res.get("base_model") or model_name
        meta_dir = mdir / base / "weights" / "openvino"
        meta = json.loads((meta_dir / "checkurai_meta.json").read_text())
        xml = mdir / base / meta.get("xml", "weights/openvino/model.xml")
        # mirror-booth models (crop mode "views" / "viewset") score the full frame: direct view + both mirrors,
        # each view on its calibration scale (eval-views "view_norm"), worst view counts
        self.views = (meta.get("crop") or {}).get("mode") in ("views", "viewset")
        if self.views:
            self.scorer = insp.MultiViewScorer(meta_dir, device=device, norm=res.get("view_norm"), mask=res.get("mask"),
                                               exposure=res.get("exposure") or False)
            self.empty = self.scorer.empty["direct"]
        else:
            self.roi = insp.load_roi()
            self.empty = insp.EmptyCheck.load(mdir / "empty_ref.png")
            self.scorer = insp.AnomalyScorer(xml, device=device)
        self.low, self.high = float(res["threshold"]["low"]), float(res["threshold"]["high"])
        self.model_name, self.device, self.force = model_name, device, force

    def spot(self, frames: list[np.ndarray]) -> tuple[bool, float]:
        if self.views:
            fracs = [self.scorer.spot(f) for f in frames]
        else:
            fracs = [self.empty.changed_fraction(self.insp.crop(f, self.roi)) for f in frames]
        frac = float(np.median(fracs))
        return frac > self.empty.frac_thr, frac

    def inspect(self, frames: list[np.ndarray]) -> Inspection:
        present, frac = self.spot(frames)
        if not present:
            return Inspection("empty", None, False, frac, len(frames), 0.0, self.low, self.high, self.model_name)
        scores, ms = [], []
        for f in frames:
            s, _, t = self.scorer(f if self.views else self.insp.crop(f, self.roi))
            scores.append(s)
            ms.append(t)
        score = float(np.median(scores))
        uncertain = self.low <= score < self.high
        verdict = "good" if score < self.low else "defect"
        if self.force:
            verdict = self.force
        f_mid = frames[int(np.argsort(scores)[len(scores) // 2])]
        mid = self.scorer.montage(f_mid) if self.views else self.insp.crop(f_mid, self.roi)
        return Inspection(verdict, round(score, 4), uncertain, round(frac, 5), len(frames), round(float(np.median(ms)), 2),
                          self.low, self.high, self.model_name, mid, [round(float(s), 4) for s in scores],
                          f_mid if self.views else None)


# --------------------------------------------------------------------------------------------
# The cell
# --------------------------------------------------------------------------------------------


@dataclass
class Stage:
    name: str
    outcome: str  # done | timeout | stalled | no_motion | stopped | failed
    load_s: float
    run_s: float
    left_s: float | None
    end_home_score: float | None
    detail: str = ""


class Abort(Exception):
    pass


class Cell:
    def __init__(self, args: argparse.Namespace, studio: StudioSession, home: HomePose, inspector: Inspector | None,
                 camera: InspectionCamera | None, run_dir: Path):
        self.a, self.s, self.home, self.inspector, self.camera, self.run_dir = args, studio, home, inspector, camera, run_dir
        self.models: dict[str, dict[str, Any]] = {}
        self.stop_requested = False

    async def resolve_models(self, stages: list[str]) -> None:
        for st in stages:
            ref = getattr(self.a, f"{st}_model")
            if not ref:
                raise SystemExit(f"--{st.replace('_', '-')}-model is needed for stage {st}")
            self.models[st] = await self.s.resolve_model(ref)
            print(f"  {st:10s} -> {self.models[st]['name']} ({self.models[st]['id']})")

    def watch_stdin(self) -> None:
        """s + Enter (or q) while a stage runs: hold the arm and stop."""
        def on_line() -> None:
            line = sys.stdin.readline().strip().lower()
            if line in ("s", "q", "stop"):
                self.stop_requested = True
                print("[cell] stop requested: holding the arm", flush=True)
        try:
            asyncio.get_running_loop().add_reader(sys.stdin.fileno(), on_line)
        except (NotImplementedError, ValueError, OSError):
            pass

    async def hide_bowl_bricks(self, name: str, model: dict[str, Any]) -> bool:
        """--hide-bowl-bricks: re-make this place model's export from a top picture taken now, with the bricks lying
        in the bowls hidden from its top camera (tools/policy_tune.py --hide-bowl-bricks; the stored top-camera warp,
        gripper settings and a painted plate are kept). True if the export changed (Studio must load it again)."""
        cmd = [str(STUDIO_PY), str(ROOT / "tools" / "policy_tune.py"), model["name"], "--hide-bowl-bricks"]
        try:  # keep a painted-in plate (not stored by policy_tune, so it has to be asked for again)
            if "paint_plate" in json.loads((Path(model["path"]) / "exports" / "openvino" / "tune.json").read_text()):
                cmd.append("--paint-plate")
        except (KeyError, OSError, ValueError):
            pass
        t = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(ROOT), stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), self.a.hide_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            print(f"[cell] {name}: hiding the bowl bricks took over {self.a.hide_timeout:.0f}s - running the export "
                  "as it is", flush=True)
            return False
        keys = ("hide bowl bricks:", "hide bricks check", "not applied", "NOT updated", "Error", "error")
        for line in out.decode(errors="replace").splitlines():
            if any(k in line for k in keys):
                print(f"[cell] {name}:   {line.strip()}", flush=True)
        ok = proc.returncode == 0
        print(f"[cell] {name}: bowl bricks {'re-made' if ok else 'NOT re-made (export as it was)'} for {model['name']} "
              f"in {time.perf_counter() - t:.1f}s", flush=True)
        return ok

    async def run_stage(self, name: str) -> Stage:
        model = self.models[name]
        reload = False  # Studio only reads the export when it loads the model: a re-made one must be loaded again
        if self.a.hide_bowl_bricks and name in ("place_good", "place_bad"):
            reload = await self.hide_bowl_bricks(name, model)
        load_s = await self.s.load_model(model["id"], self.a.backend, self.a.device, force=reload)
        det = DoneDetector(self.home, still_s=self.a.still_s, still_range=self.a.still_range, stall_s=self.a.stall_s,
                           hold_score=self.a.hold_score)
        task = getattr(self.a, f"{name}_task") or TASKS[name]
        print(f"[cell] {name}: model {model['name']} (load {load_s:.1f}s), task '{task}'", flush=True)
        t_run, t_run_mono = time.perf_counter(), time.monotonic()
        await self.s.run_model(task, self.a.hz, self.a.execution, self.a.request_threshold, self.a.grace_s)
        outcome, detail, last_t = "timeout", "", 0.0
        # Re-planning often from HOME never starts: the demos wait ~1 s there, so every fresh chunk begins with
        # that wait again. Start at --request-threshold; switch to --moving-threshold once the arm is under way.
        switch = self.a.moving_threshold
        try:
            while True:
                await asyncio.sleep(0.01)
                if self.stop_requested:
                    outcome, detail = "stopped", "operator"
                    break
                if self.s.closed.is_set():
                    outcome, detail = "failed", "websocket closed"
                    break
                bad = [p for p in self.s.failed_parts() if p.split(":")[0] in ("control", "model", "environment",
                                                                                 "environment/follower")]
                if bad:
                    outcome, detail = "failed", "; ".join(bad)
                    break
                if self.s.joints_t != last_t:
                    last_t = self.s.joints_t
                    st = det.update(last_t, self.s.q(self.home.names))
                    if switch is not None and st == "moving":
                        await self.s.run_model(task, self.a.hz, self.a.execution, switch, 0.0)
                        print(f"[cell] {name}: under way after {time.perf_counter() - t_run:.1f}s; "
                              f"re-planning at {switch}", flush=True)
                        switch = None
                    if st == "done":
                        outcome = "done"
                        break
                    if st == "stalled" and self.a.stall_abort:
                        outcome, detail = "stalled", "still away from HOME"
                        break
                    if st == "start" and time.perf_counter() - t_run > self.a.start_timeout:
                        outcome, detail = "no_motion", "the arm never left HOME"
                        break
                if time.perf_counter() - t_run > self.a.stage_timeout:
                    break
        finally:
            await self.s.hold()
        q = self.s.q(self.home.names)
        stage = Stage(name, outcome, round(load_s, 2), round(time.perf_counter() - t_run, 2),
                      None if det.t_left is None else round(det.t_left - t_run_mono, 2),
                      None if q is None else round(self.home.score(q), 2), detail)
        print(f"[cell] {name}: {outcome} in {stage.run_s:.1f}s {detail}", flush=True)
        return stage

    async def rehome(self) -> bool:
        """Release the arm from Studio (its torque stays on), drive it HOME over its own bus with
        tools/go_home.py, and load the environment again."""
        print("[cell] bringing the arm HOME to try again from the top", flush=True)
        await self.s.command("unload_environment", timeout=60)
        proc = await asyncio.create_subprocess_exec(sys.executable, str(ROOT / "tools" / "go_home.py"))
        rc = await proc.wait()
        self.s.statuses.clear()
        t = self.s.joints_t
        await self.s.load_environment(self.a.environment)
        for _ in range(100):
            if self.s.joints_t != t:
                break
            await asyncio.sleep(0.1)
        q = self.s.q(self.home.names)
        deadline = time.monotonic() + 5.0
        while rc == 0 and (q is None or not self.home.near(q)) and time.monotonic() < deadline:
            await asyncio.sleep(0.2)  # Studio's first readings after a reload can still be the pre-HOME pose
            q = self.s.q(self.home.names)
        ok = rc == 0 and q is not None and self.home.near(q)
        print(f"[cell] {'arm is HOME' if ok else 'could not bring the arm HOME'}", flush=True)
        return ok

    async def run_stage_tries(self, name: str) -> list[Stage]:
        """run_stage; if it times out, stalls or the follower bus fails (e.g. the arm keeps trying low down
        after the brick slipped), bring the arm HOME and start the stage again, up to --rehome-retries times."""
        stages = [await self.run_stage(name)]
        for _ in range(self.a.rehome_retries):
            if stages[-1].outcome not in ("timeout", "stalled", "failed") or self.stop_requested:
                break
            if not await self.rehome():
                break
            if name == "pick" and self.brick_on_spot_at_home():
                print(f"[cell] pick {stages[-1].outcome}, but the brick is on the spot: going on to inspect it",
                      flush=True)
                break
            stages.append(await self.run_stage(name))
        return stages

    def spot(self) -> tuple[bool, float]:
        if self.camera is None or self.inspector is None:
            return True, -1.0
        return self.inspector.spot(self.camera.grab(3))

    def brick_on_spot_at_home(self) -> bool:
        """After a pick that did not end at HOME (the policy hovered over the spot until the timeout) and a re-home:
        the arm is HOME and the spot camera sees the brick, so the pick did its job."""
        if self.camera is None or self.inspector is None:
            return False
        q = self.s.q(self.home.names)
        return q is not None and self.home.near(q) and self.spot()[0]

    async def reinspect(self, rec: dict[str, Any], first: Inspection) -> Inspection:
        """An uncertain verdict gets one more look before the cell decides: fresh frames, same model, same thresholds.
        The brick has not moved (same pose on the spot), so this mostly averages out frame noise (exposure, flicker,
        the arm's shadow leaving); it cannot reveal a face the cameras don't see. A real re-grasp would need a policy
        that picks the brick off the spot, which we don't have. Still uncertain -> defect bowl, as before (fail-safe).
        rec["inspection"] then holds the deciding look, plus looks=2 and the other look."""
        n = int(getattr(self.a, "reinspect_frames", 5))
        print(f"[cell] uncertain ({first.score}); inspecting again with {n} fresh frames", flush=True)
        await asyncio.sleep(float(getattr(self.a, "reinspect_wait", 1.0)))
        t = time.perf_counter()
        again = self.inspector.inspect(self.camera.grab(n))
        second = {**again.log(), "total_ms": round((time.perf_counter() - t) * 1000, 1)}
        first_rec = rec["inspection"]
        if again.verdict == "empty":  # the brick was not seen this time (should not happen): keep the first look
            rec["inspection"] = {**first_rec, "looks": 2, "decided_by": "first", "reinspection": second}
            print("[cell] re-inspection: no brick seen; keeping the first inspection (uncertain -> defect)", flush=True)
            return first
        rec["inspection"] = {**second, "looks": 2, "decided_by": "second", "first_inspection": first_rec}
        print(f"[cell] re-inspection: {again.verdict} {again.score}{' (still uncertain)' if again.uncertain else ''}",
              flush=True)
        return again

    def save_evidence(self, cycle: int, insp: Inspection) -> dict[str, str]:
        import cv2

        out = {}
        if insp.crop is not None:
            p = self.run_dir / f"cycle_{cycle:03d}_inspect.jpg"
            img = insp.crop.copy()
            color = (60, 180, 60) if insp.verdict == "good" else (40, 60, 220)
            label = f"{insp.verdict.upper()}{' ?' if insp.uncertain else ''}  {insp.score:.1f}"
            cv2.putText(img, label, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
            cv2.imwrite(str(p), img)
            out["inspect"] = p.name
        if insp.frame is not None:  # the full frame, so a new inspection model can be re-scored on tonight's cycles
            p = self.run_dir / f"cycle_{cycle:03d}_frame.jpg"
            cv2.imwrite(str(p), insp.frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            out["frame"] = p.name
        if "top" in self.s.frames:
            p = self.run_dir / f"cycle_{cycle:03d}_top.jpg"
            p.write_bytes(self.s.frames["top"][1])
            out["top"] = p.name
        return out

    async def cycle(self, i: int) -> dict[str, Any]:
        rec: dict[str, Any] = {"cycle": i, "started": now_iso(), "stages": []}
        t0 = time.perf_counter()
        try:
            q = self.s.q(self.home.names)
            if q is not None and not self.home.near(q) and self.home.score(q) <= getattr(self.a, "hold_score", 1.4):
                # the last stage counted as done while resting just off HOME (--hold-score): settle it HOME first
                await self.rehome()
                q = self.s.q(self.home.names)
            if q is None or not self.home.near(q):
                raise Abort(f"arm is not at HOME (score {self.home.score(q) if q is not None else 'n/a'}); "
                            "bring it HOME first")
            present, frac = self.spot()
            rec["spot_before"] = {"brick": present, "fraction": round(frac, 5)}
            if not present or self.camera is None:
                for attempt in range(1 + self.a.pick_retries):
                    sts = await self.run_stage_tries("pick")
                    rec["stages"] += [asdict(x) for x in sts]
                    st = sts[-1]
                    if st.outcome != "done" and not (st.outcome in ("timeout", "stalled")
                                                     and self.brick_on_spot_at_home()):
                        raise Abort(f"pick {st.outcome}")
                    present, frac = self.spot()
                    rec["spot_after_pick"] = {"brick": present, "fraction": round(frac, 5), "attempt": attempt + 1}
                    if present:
                        break
                    print("[cell] spot is empty after the pick; retrying", flush=True)
                if not present:
                    raise Abort("no brick on the spot after the pick retries (pick zone empty?)")
            # inspect
            if self.inspector is not None and self.camera is not None:
                t = time.perf_counter()
                insp = self.inspector.inspect(self.camera.grab(self.a.frames))
                rec["inspection"] = {**insp.log(), "total_ms": round((time.perf_counter() - t) * 1000, 1)}
                if insp.uncertain and insp.verdict != "empty" and getattr(self.a, "reinspect_uncertain", True):
                    insp = await self.reinspect(rec, insp)
                rec["evidence"] = self.save_evidence(i, insp)
                verdict = insp.verdict
            else:
                verdict = self.a.force_verdict or "defect"
                rec["inspection"] = {"verdict": verdict, "forced": True}
            print(f"[cell] verdict: {verdict} {rec['inspection'].get('score', '')}"
                  f"{' (uncertain)' if rec['inspection'].get('uncertain') else ''}", flush=True)
            if verdict == "empty":
                raise Abort("inspection saw an empty spot")
            place = "place_good" if verdict == "good" else "place_bad"
            for attempt in range(1 + self.a.place_retries):
                sts = await self.run_stage_tries(place)
                rec["stages"] += [asdict(x) for x in sts]
                st = sts[-1]
                if st.outcome != "done":
                    raise Abort(f"{place} {st.outcome}")
                present, frac = self.spot()
                rec["spot_after_place"] = {"brick": present, "fraction": round(frac, 5), "attempt": attempt + 1}
                if not present or self.camera is None:
                    break
                print("[cell] brick still on the spot after the place (missed grasp); retrying", flush=True)
            rec["result"] = "sorted" if not present else "brick still on the spot"
        except Abort as e:
            rec["result"] = f"aborted: {e}"
        rec["cycle_s"] = round(time.perf_counter() - t0, 2)
        return rec


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------


def load_or_learn_home(args: argparse.Namespace) -> HomePose:
    if Path(args.home).exists():
        return HomePose.load(args.home)
    if not args.home_dataset:
        raise SystemExit(f"{args.home} is missing; run learn-home first or pass --home-dataset")
    home = learn_home(args.home_dataset)
    Path(args.home).write_text(json.dumps(home.to_json(), indent=1))
    return home


def print_home(home: HomePose, q: np.ndarray | None = None) -> None:
    print(f"{'joint':14s} {'home':>8s} {'tol':>6s}" + ("  {:>8s} {:>6s}".format("now", "dev/tol") if q is not None else ""))
    for j, n in enumerate(home.names):
        line = f"{n:14s} {home.q[j]:8.2f} {home.tol[j]:6.2f}"
        if q is not None:
            line += f"  {q[j]:8.2f} {abs(q[j] - home.q[j]) / home.tol[j]:6.2f}"
        print(line)


def cmd_learn_home(args: argparse.Namespace) -> None:
    home = learn_home(args.dataset)
    Path(args.home).parent.mkdir(parents=True, exist_ok=True)
    Path(args.home).write_text(json.dumps(home.to_json(), indent=1))
    print_home(home)
    print(f"saved {args.home}")


def cmd_replay(args: argparse.Namespace) -> None:
    home = load_or_learn_home(args)
    res = replay(args.dataset, home, {"still_s": args.still_s, "still_range": args.still_range, "stall_s": args.stall_s})
    for r in res["rows"]:
        print(r)
    print({k: v for k, v in res.items() if k != "rows"})


async def cmd_dry_run(args: argparse.Namespace) -> None:
    """Never loads an environment, so it never touches the robot or the cameras."""
    home = load_or_learn_home(args)
    sessions = await list_sessions(args.url)
    print(f"live runtime sessions: {len(sessions)}")
    for s in sessions:
        print(f"  {s['session_id']}: project {s['project_name']!r}, environment {s['environment_name']!r}, "
              f"robots {len(s['robot_ids'])}, cameras {len(s['camera_ids'])}")
    async with StudioSession(args.url, args.project, args.hz) as studio:
        print("our session state:", json.dumps(studio.state))
        env = next((e for e in await studio.get(f"/api/projects/{args.project}/environments")
                    if args.environment in (e["id"], e["name"])), None)
        robot_ids = {r["robot_id"] for r in (env or {}).get("robots", [])}
        busy = [s for s in sessions if robot_ids & set(s["robot_ids"])]
        print(f"environment {args.environment!r}: {'found' if env else 'MISSING'}; follower "
              f"{'held by session ' + busy[0]['session_id'] + ' (close that page before run)' if busy else 'free'}")
        cell = Cell(args, studio, home, None, None, Path("."))
        stages = [s for s in ("pick", "place_good", "place_bad") if getattr(args, f"{s}_model")]
        await cell.resolve_models(stages)
    print_home(home)


async def cmd_run(args: argparse.Namespace) -> None:
    home = load_or_learn_home(args)
    run_dir = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    camera = inspector = None
    if not args.no_inspection:
        camera = InspectionCamera(args.camera_name, args.camera_device).start()
        if not args.no_align:
            from tools.frame_align import FrameAligner

            camera.aligner = FrameAligner.try_load()
            if camera.aligner is not None:
                camera.aligner.align(camera.grab(1)[0])
                print(f"inspection camera vs the anomaly photos: {camera.aligner.info} (frames are aligned)")
        if (ROOT / "models" / "classifier" / args.anomaly_model).is_dir():  # supervised defect classifier
            from tools.brick_classifier import BrickClassifierInspector

            inspector = BrickClassifierInspector(args.anomaly_model, args.anomaly_device, args.force_verdict)
        else:
            inspector = AnomalibInspector(args.anomaly_model, args.anomaly_device, args.force_verdict)
        print(f"inspection: {camera.device}, model {args.anomaly_model} on {args.anomaly_device}, "
              f"threshold {inspector.low:.2f} / band {inspector.high:.2f}")
    log = (run_dir / "cycles.jsonl").open("a")
    try:
        async with StudioSession(args.url, args.project, args.hz) as studio:
            cell = Cell(args, studio, home, inspector, camera, run_dir)
            stages = [args.only] if args.only else ["pick", "place_good", "place_bad"]
            await cell.resolve_models(stages)
            print(f"loading environment {args.environment!r} ...", flush=True)
            await studio.load_environment(args.environment)
            for _ in range(100):
                if studio.joints:
                    break
                await asyncio.sleep(0.1)
            q = studio.q(home.names)
            print_home(home, q)
            if q is None:
                raise SystemExit("no joint readings from the follower")
            if not home.near(q) and not args.allow_off_home:
                raise SystemExit("the follower is not at HOME; bring it HOME (teleop) and run again")
            meta = {"run": run_dir.name, "args": {k: v for k, v in vars(args).items() if k != "func"},
                    "home": home.to_json(), "models": {k: v["name"] for k, v in cell.models.items()}}
            (run_dir / "run.json").write_text(json.dumps(meta, indent=1, default=str))
            if not args.yes:
                ans = input("Person at the 12 V switch and the cell clear? Type go: ").strip().lower()
                if ans != "go":
                    raise SystemExit("not started")
            cell.watch_stdin()
            print("running; type s + Enter to hold the arm and stop", flush=True)
            try:
                if args.only:
                    sts = await cell.run_stage_tries(args.only)
                    rec = {"cycle": 0, "only": args.only, "stages": [asdict(x) for x in sts], "started": now_iso()}
                    log.write(json.dumps(rec, default=str) + "\n")
                    log.flush()
                else:
                    for i in range(args.cycles):
                        rec = await cell.cycle(i)
                        log.write(json.dumps(rec, default=str) + "\n")
                        log.flush()
                        print(f"[cell] cycle {i}: {rec['result']} in {rec['cycle_s']:.1f}s", flush=True)
                        if cell.stop_requested or rec["result"].startswith("aborted"):
                            break
            finally:
                try:
                    await asyncio.shield(studio.hold())
                except Exception as e:  # noqa: BLE001 - best effort on the way out
                    print(f"hold on exit failed: {e}")
                if args.keep_session:
                    input("Arm held. Press Enter to close the session (bring the arm HOME first if it is up): ")
    finally:
        log.close()
        if camera is not None:
            camera.stop()
    print(f"log: {run_dir / 'cycles.jsonl'}")


def build_parser() -> argparse.ArgumentParser:
    cfg = json.loads(DEFAULT_CELL_CONFIG.read_text()) if DEFAULT_CELL_CONFIG.exists() else {}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=cfg.get("url", DEFAULT_URL), help="Studio backend")
    ap.add_argument("--project", default=cfg.get("project"), help="Studio project id")
    ap.add_argument("--environment", default=cfg.get("environment", "pick"), help="environment name or id")
    ap.add_argument("--home", default=cfg.get("home", str(DEFAULT_HOME)), help="HOME pose json")
    ap.add_argument("--home-dataset", default=cfg.get("home_dataset"), help="learn HOME from this dataset if needed")
    for st in ("pick", "place_good", "place_bad"):
        ap.add_argument(f"--{st.replace('_', '-')}-model", dest=f"{st}_model", default=cfg.get(f"{st}_model"),
                        help=f"model name or id for {st}")
        ap.add_argument(f"--{st.replace('_', '-')}-task", dest=f"{st}_task", default=cfg.get(f"{st}_task"),
                        help=f"task string for {st} (default: the dataset task)")
    ap.add_argument("--backend", default=cfg.get("backend", "openvino"))
    ap.add_argument("--device", default=cfg.get("device", "GPU"), help="OpenVINO device for the policies")
    ap.add_argument("--hz", type=float, default=cfg.get("hz", 30.0), help="action rate (and observation rate)")
    ap.add_argument("--execution", choices=["sync", "async"], default=cfg.get("execution", "sync"))
    ap.add_argument("--request-threshold", type=float, default=cfg.get("request_threshold", 0.5),
                    help="ask for the next chunk when this fraction of the current one is left (0.5: every 50 steps)")
    ap.add_argument("--moving-threshold", type=float, default=cfg.get("moving_threshold"),
                    help="once the arm has left HOME, re-plan at this threshold instead (e.g. 0.8: every 20 steps)")
    ap.add_argument("--grace-s", type=float, default=cfg.get("grace_s", 3.0))
    ap.add_argument("--hide-bowl-bricks", action="store_true", default=cfg.get("hide_bowl_bricks", False),
                    help="before each place stage: re-make its model so its top camera sees the bricks already "
                         "lying in the bowls as empty bowl (tools/policy_tune.py --hide-bowl-bricks, ~3 s) and "
                         "load it again")
    ap.add_argument("--hide-timeout", type=float, default=cfg.get("hide_timeout", 30.0),
                    help="--hide-bowl-bricks: give up on the re-make after this many seconds (runs as it was)")
    ap.add_argument("--still-s", type=float, default=cfg.get("still_s", 0.7))
    ap.add_argument("--still-range", type=float, default=cfg.get("still_range", 1.5))
    ap.add_argument("--stall-s", type=float, default=cfg.get("stall_s", 4.0))
    ap.add_argument("--hold-score", type=float, default=cfg.get("hold_score", 1.4),
                    help="a stage counts as done once the arm stays within this many x tolerance of HOME for 1.5 s "
                         "(act-pick-v6-h100 rests at ~2.4x: elbow / wrist flex a little off HOME)")
    ap.add_argument("--stall-abort", action=argparse.BooleanOptionalAction, default=cfg.get("stall_abort", True))
    ap.add_argument("--stage-timeout", type=float, default=cfg.get("stage_timeout", 40.0))
    ap.add_argument("--start-timeout", type=float, default=cfg.get("start_timeout", 10.0),
                    help="seconds the arm may stay at HOME after a stage starts")
    ap.add_argument("--anomaly-model", default=cfg.get("anomaly_model", "patchcore_384"))
    ap.add_argument("--anomaly-device", default=cfg.get("anomaly_device", "GPU"))
    ap.add_argument("--frames", type=int, default=cfg.get("frames", 9), help="inspection frames per brick (median)")
    ap.add_argument("--reinspect-uncertain", action=argparse.BooleanOptionalAction, default=cfg.get("reinspect_uncertain", True),
                    help="an uncertain verdict takes one more look (fresh frames, same pose) before it decides; "
                         "still uncertain -> defect bowl")
    ap.add_argument("--reinspect-frames", type=int, default=cfg.get("reinspect_frames", 5),
                    help="frames for the re-inspection (median)")
    ap.add_argument("--reinspect-wait", type=float, default=cfg.get("reinspect_wait", 1.0),
                    help="seconds to wait before the re-inspection")
    ap.add_argument("--camera-name", default=cfg.get("camera_name", "Webcam Audio"))
    ap.add_argument("--camera-device", default=cfg.get("camera_device"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("learn-home", help="learn HOME from a dataset")
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=cmd_learn_home)
    p = sub.add_parser("replay", help="run the done detector over a dataset's episodes")
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=cmd_replay)
    p = sub.add_parser("dry-run", help="connect and report; never touches the robot")
    p.set_defaults(func=cmd_dry_run)
    p = sub.add_parser("run", help="drive the cell")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--only", choices=["pick", "place_good", "place_bad"], help="run one stage once")
    p.add_argument("--pick-retries", type=int, default=1)
    p.add_argument("--place-retries", type=int, default=1, help="run the place again if the brick is still on the spot")
    p.add_argument("--rehome-retries", type=int, default=1,
                   help="after a timeout / stall / bus error: drive the arm HOME and start the stage again")
    p.add_argument("--no-inspection", action="store_true", help="skip the camera and anomaly model")
    p.add_argument("--no-align", action="store_true",
                   help="do not map inspection frames onto the anomaly photos' geometry (tools/frame_align.py)")
    p.add_argument("--force-verdict", choices=["good", "defect"], help="override the verdict (testing)")
    p.add_argument("--allow-off-home", action="store_true")
    p.add_argument("--keep-session", action="store_true", help="wait for Enter before releasing the robot")
    p.add_argument("--yes", action="store_true", help="skip the operator confirmation")
    p.set_defaults(func=cmd_run)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd in ("dry-run", "run") and not args.project:
        raise SystemExit("--project is needed (or put it in config/cell.json)")
    out = args.func(args)
    if asyncio.iscoroutine(out):
        asyncio.run(out)


if __name__ == "__main__":
    main()
