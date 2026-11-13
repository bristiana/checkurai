# Intel stack brief: Physical AI Studio, OpenVINO Physical AI, Anomalib, SO-101 (as of 13 Nov 2026)

From a web research pass on primary sources (GitHub repos and official docs). Items marked **unverified** need checking on the station laptop.

## The 5 things that matter most
1. **The laptops should come preinstalled, on Ubuntu 24.04.** NPU and Arc drivers, conda env `intel_dev_env`, Physical AI Studio, OpenVINO 2026.3, Anomalib 2.6.0, and LeRobot on PyTorch XPU. **Run `python verify_stack.py` first.** Source: github.com/open-edge-platform/robotics-ai-suite/blob/main/docs/user-guide/resources/hackathon_resources.md
2. **The Studio UI only runs policies on CPU or GPU, not the NPU** (the docs say NPU is "not yet verified"). To use the **NPU**, run the exported model through the **Physical AI Runtime** (`pip install physicalai`) with `device: NPU`.
3. **Python versions may clash.** `intel_dev_env` is Python 3.11, but Studio 0.3, the Runtime plugins and LeRobot 0.7 all want 3.12 or later. The LeRobot in the env is probably older, so its CLI may differ from the current docs. Keep Studio and the Runtime in their own 3.12 envs if needed. **Unverified; check on the laptop.**
4. **Policies:** use **ACT** as the fast, reliable baseline (2 GB VRAM) and **SmolVLA** as the VLA (3 GB; it takes a language instruction, so it can be told which bowl). Pi0.5 needs 40 GB to train.
5. **Camera sharing is built in.** `physicalai.capture.SharedCamera` lets one process own a camera while the policy and the inspector both read its frames. Intel explicitly asked for this idea.

## Physical AI Studio (github.com/open-edge-platform/physical-ai-studio, v0.3.0, Apache-2.0)
- **Parts:** a web UI, a Python library (`physicalai-train` CLI) and the separate Runtime. Linux only in the docs.
- **Start it on the hackathon laptop** (2 terminals):
  - `cd ~/physical-ai-studio/application/backend && ./run.sh`
  - `cd ~/physical-ai-studio/application/ui && nvm use && npm run start`
  - Then open **http://localhost:3000**. Over SSH: `ssh -L 3000:localhost:3000 intel`.
  - pip alternative: `physicalai-studio serve` on :7860.
- **Hardware:**
  - **SO-101 leader and follower are built in**, with a wizard for motor setup, calibration and verification.
  - Cameras: USB, **RealSense**, Basler, IP. Recommended 640×480 at 30 fps.
- **Policies:** ACT, SmolVLA, Pi0.5, MolmoAct2, RLDX-1, XR0, Cosmos3. No pi0 or GR00T in Studio (GR00T N1.7 is a separate Robotics AI Suite pipeline). SnapFlow distillation (SmolVLA, Pi0.5) is advertised as "up to 2x faster".
- **UI flow:**
  - Project → Robots → Cameras → Environment (follower + leader + cameras) → Dataset.
  - Record episodes (→ starts or accepts, ← discards), then Train, then Run model.
  - Datasets are LeRobot v3 format.
  - Studio recommends **at least 50 episodes**.
- **Training on cloud GPUs:** Settings → Training Targets. Choose a Trainer URL, an SSH host with Docker (needs passwordless sudo for auto-setup) or AWS. Studio uploads a dataset snapshot and downloads the model.
- **Export:**
  - **Automatic after training:** PyTorch, OpenVINO, ONNX and ExecuTorch. An interrupted run still exports.
  - **Runtime export** (on the OpenVINO card) gives a zip with the model, `runtime.yaml` and a README. Run it with `physicalai run --config runtime.yaml`.
  - CLI: `physicalai export --policy physicalai.policies.ACT --ckpt_path …ckpt --backend openvino --output_dir exports/act`
  - INT8 weights-only compression: hook `compress_weights_openvino_int8_sym` (needs the `[nncf]` extra).
- **Gotchas:**
  - SmolVLA and Pi0.5 need an **HF token** (Settings → General → Hugging Face).
  - Recording needs FFmpeg/PyAV.
  - **Camera names, order and resolution must be identical when recording and when deploying.**
  - chunk_size, hidden_dim and camera_names must match between training and inference, or you get a shape-mismatch error.

## Physical AI Runtime = "OpenVINO Physical AI" (github.com/openvinotoolkit/physicalai, `pip install physicalai`, v0.4.0, preview)
- Load a model: `InferenceModel(export_dir, backend="openvino", device="CPU"|"GPU"|"NPU"|"AUTO")`, then `.select_action(obs)`.
- `RobotRuntime` takes a `PolicySource` or `TeleopSource`, with Sync, Async or RTC execution. `PolicySource.reset()` / `.warmup()` switch tasks without reconnecting the hardware, which fits the pick → place phases.
- Cameras: UVC, `physicalai[realsense]` (RGB-D), `SharedCamera`. SO-101 plugin. `SharedRobot` over Zenoh (no auth in remote mode).
- **`RerunCallback` dashboard:** CPU/RAM, inference latency, GPU stats. Use it for the demo.
- **`InferenceLatencyBenchmark`:** mean, median, min and max latency per chunk. Use it for the CPU vs GPU vs NPU numbers.
- Run: `physicalai run --config runtime.yaml --run.duration_s=60`
- There are no published ACT or SmolVLA latencies on the Panther Lake NPU, and the NPU path is unvalidated in Studio. Benchmark it yourselves.

## Robotics AI Suite (github.com/open-edge-platform/robotics-ai-suite; Ubuntu 24.04 + ROS 2 Jazzy)
- No SO-101-specific pipeline; that path is Studio plus the Runtime.
- Useful pieces:
  - The ACT/OpenVINO pipeline (simulation, ALOHA).
  - Pi0.5+RTC and GR00T N1.7 OpenVINO pipelines.
  - OpenVINO YOLOv8/v12 detection (finding bricks in clutter from the top camera).
  - The pre-release "Stationary Robot Vision & Control" pick-and-place framework.
- **Release 2026.2 adds Core Ultra Series 3 core-pinning and frequency-locking scripts.** They reduce latency jitter and count as "Intel integration" for judging.

## Anomalib (2.6.0 installed with [xpu,openvino]; 2.7.0 is the latest)
- **Model choice:**
  - **PatchCore** by default: needs only good images, no gradient training, very accurate.
  - **PaDiM** as a lighter fallback.
  - **EfficientAD** for the lowest latency, likely the best fit for the NPU.
  - Zero/few-shot options: WinCLIP, AnomalyDINO, FoundAD.
  - **Unverified:** whether PatchCore compiles on the NPU. If it doesn't, use PaDiM or EfficientAD on the NPU.
- **Data (rule of thumb):**
  - About 100–200 **good** images at the **fixed inspection pose** (vary orientation and colour), plus 20–50 defective images to set the threshold.
  - The biggest accuracy lever is the same pose against a plain background.
  - Synthetic anomalies are possible with `TestSplitMode.SYNTHETIC`.
- **Code:**
```python
from anomalib.data import Folder; from anomalib.models import Patchcore; from anomalib.engine import Engine
from anomalib.deploy import ExportType, OpenVINOInferencer
dm = Folder(name="duplo", root="data/duplo", normal_dir="good", abnormal_dir="defective")
model, engine = Patchcore(), Engine(); engine.fit(model=model, datamodule=dm)
engine.export(model, export_type=ExportType.OPENVINO, export_root="exported")
inf = OpenVINOInferencer(path="exported/weights/openvino/model.xml", device="NPU")  # check the path
pred = inf.predict(frame); pred.pred_score
```
- **Benchmark:** `benchmark_app -m model.xml -d NPU -hint latency` (repeat with CPU and GPU). Anomaly detection won't be the bottleneck: presenting the brick takes over a second.

## SO-101 + LeRobot practical notes
- **Arm controllers:**
  - Both are CH343 (`1a86:55d3`), seen as `/dev/ttyACM*`.
  - **Leader serial 5B90148356, follower 5AE7045345** (from our scan). Use `/dev/serial/by-id/…` instead of ttyACM numbers.
  - You need to be in the `dialout` group (or have Intel set up permissions) to open the ports.
- **Commands (LeRobot 0.7.x; older versions differ):**
  - `lerobot-find-port`, `lerobot-calibrate --robot.type=so101_follower --robot.port=… --robot.id=follower`
  - `lerobot-calibrate --teleop.type=so101_leader --teleop.port=… --teleop.id=leader`, `lerobot-find-cameras opencv|realsense`
  - `lerobot-record … --robot.cameras="{ wrist: {type: opencv, …}, top: {type: intelrealsense, serial_number_or_name: '<SN>', …}}" --dataset.num_episodes=50 --dataset.push_to_hub=false`
  - **Use the same `id` everywhere** (calibration is stored by id).
- **Demos:**
  - LeRobot: ≥50 episodes, about 10 per position.
  - SmolVLA authors: **50 over 5 positions worked; 25 was not enough**. 20k steps takes about 4 h on an A100.
  - ACT converges in 5–10 epochs, so don't run the default 100k steps.
- **Failure modes:**
  - **Never swap power supplies:** leader 5 V, follower 12 V. Our scan confirmed 5.4 V and 12.0 V.
  - **The gripper can burn out** if it squeezes for long; turn torque off between runs.
  - Recalibrate if the leader and follower diverge, and never reuse another arm's calibration.
  - Check the recording cadence report: a loop that actually ran at 25 Hz still gets stored as 30 Hz.

## Core Ultra X9 388H (Panther Lake), probably our chip (unverified)
- **CPU:** 16 cores (4P + 8E + 4LP-E), up to 5.1 GHz, 25 W base / 80 W turbo.
- **iGPU:** Arc B390 (12 Xe3 cores), about 120 INT8 TOPS.
- **NPU 5:** 50 TOPS. Platform total 180 TOPS.
- **NPU on Linux:** needs kernel 6.6 or later and shows up as `/dev/accel/accel0`. It wants **static shapes**; set `CACHE_DIR` to avoid slow first compiles.

## Weekend plan (fits the judging)
- **Architecture:** a Python state machine.
  1. Policy A picks the brick and presents it at the inspection pose.
  2. Anomalib checks the inspection-camera frame (shared camera).
  3. Place in the good or defective bowl: a separate ACT policy per bowl, or SmolVLA with the task string switched plus `reset()`.
  4. If the score is uncertain or no brick is seen, re-grasp and re-inspect.
- **Fri night:**
  - Run `verify_stack.py` and note the versions.
  - Calibrate both arms in Studio.
  - **Fix the cameras and lighting rigidly; tape the pick zone, inspection pose and bowls.**
  - Record 5 episodes and push them through train → export → `physicalai run` once, to prove the pipeline end to end.
  - Start capturing anomaly images.
- **Sat morning:**
  - Record 50–60 clean demos (about 6 positions × 10, plus 10–20 shifted or cluttered); two operators alternating, about 1 h.
  - Train ACT and SmolVLA in the cloud.
  - Fit PatchCore and benchmark it on CPU, GPU and NPU.
- **Sat afternoon:**
  - Run 20 trials; record 15–20 extra demos targeting the failures; retrain.
  - Train ACT on 10, 25 and 50 episodes to get the **"how few demos are enough" curve**.
- **Sat night:**
  - Try the policy on GPU with anomaly on NPU, then the reverse.
  - INT8 compression; SnapFlow; AsyncExecution; core-pinning scripts.
- **Sun morning:** freeze the code. Run 30 or more bricks (mixed, shifted, cluttered) and rehearse with the Rerun dashboard.
- **Show these numbers:**
  - Success rate and confusion matrix.
  - Cycle time (mean and p95) and control-loop Hz.
  - Latency per device.
  - Package power and temperature: `sudo turbostat --show PkgWatt,GFXWatt,PkgTmp -i 1` (needs sudo; ask Intel).
  - CPU headroom with offloading versus without.
  - The demos-versus-success curve.
