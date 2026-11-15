# CheckurAI

Junction 2026, Intel Physical AI Challenge: an SO-101 arm picks Duplo bricks onto an inspection spot, an
Anomalib model judges them from three views (direct + two mirrors), and the arm drops each brick in the good
or the defect bowl. Everything runs on the Intel Core Ultra station laptop.

Licence: [MIT](LICENSE). Third-party software: [THIRD_PARTY.md](THIRD_PARTY.md).

## How it works
- **Record, train, export:** demos recorded with the leader arm in Physical AI Studio; ACT policies
  (`pick`, `place_good`, `place_bad`) trained through Studio on cloud GPUs and exported to OpenVINO.
- **Run:** `tools/orchestrator.py` drives the arm only through Studio's runtime session websocket
  (`load_environment`, `load_model` with the OpenVINO backend, `set_control`), so the policies run in the
  `physicalai` runtime on the device set in `config/cell.json` (currently the iGPU).
- **Inspect:** `tools/anomaly_train.py` trains Anomalib PatchCore on photos of good bricks and exports it to
  OpenVINO; `tools/inspection.py` scores the three views of the inspection camera (median over several
  frames) and returns good / defect / uncertain. An uncertain brick is inspected once more (fresh frames,
  same pose, added at the end of the event); still uncertain → the defect bowl.
- **Checks and retries:** the inspection camera checks the spot before the pick, after the pick (empty →
  retry the pick once) and after the place (brick still there → retry once); a stage that times out or stalls
  sends the arm HOME and reruns once. Type `s` + Enter to hold the arm and stop.
- **Policy fixes baked into the OpenVINO graph** (`tools/policy_tune.py`): a wider gripper opening, and a
  top-camera warp (ArUco markers) that undoes camera drift since the model's demos. Optional, off by default:
  hide the bricks already lying in the bowls from the place policy, paint the pick plate back in, follow a moved
  plate.

## Run the cell (on the station laptop)
1. Physical AI Studio backend running (http://localhost:7860), with the station's robot and cameras set up
   (`tools/studio_setup.py`) and the models named in `config/cell.json` exported to OpenVINO. Close Studio's
   record / inference pages: the cell program holds the robot.
2. Check without moving: `python tools/orchestrator.py dry-run`
3. Run N bricks with the live jury feed (same arguments as `tools/orchestrator.py`):
   `python tools/cell_live.py run --cycles N` (one stage only: `--only pick`; no camera: `--no-inspection`)
4. Jury dashboard: `python tools/dashboard.py --page dashboard/live.html` → http://127.0.0.1:8770
   (from another machine: `ssh -N -L 8770:localhost:8770 intel`).

Use the Python environment that has Anomalib, OpenVINO and aiohttp (`intel_dev_env` on the station);
`tools/policy_tune.py` runs with Studio's own venv. Each run writes `data/cell_runs/<time>/` (gitignored).

## What's in here
| | |
|---|---|
| `tools/orchestrator.py`, `tools/cell_live.py` | the cell program; the live feed for the dashboard |
| `tools/inspection.py`, `tools/anomaly_train.py`, `tools/capture_inspection.py`, `tools/frame_align.py` | inspection camera, Anomalib training/export/evaluation, photo capture, frame alignment |
| `tools/policy_tune.py`, `tools/bowl_check.py`, `tools/bowl_follow.py`, `tools/bowl_react.py` | OpenVINO graph fixes; where the bowls/plate stood in each policy's demos vs now (ArUco), analyses |
| `tools/dashboard.py`, `tools/scene_overlay.py`, `dashboard/` | jury dashboard (`live.html`; `index.html` = first version, `mock_server.py` = demo data) |
| `tools/bench_devices.py`, `tools/power_log.py`, `tools/power_report.py` | CPU/GPU/NPU latency bench; power, temperature and accelerator-load log; numbers per cycle from that log |
| `tools/calib_*.py`, `tools/studio_setup.py`, `tools/studio_push_calib.py`, `final_calib.sh` | arm calibration (LeRobot) and Studio registration |
| `tools/episode_check.py`, `tools/dataset_merge.py`, `tools/scene_history.py` | demo dataset checks, merging, camera/bowl position per demo episode |
| `tools/cam_lock.py`, `tools/cam_guard.py`, `tools/cell_check.py`, `tools/aim_view.py` | lock camera exposure/white balance/focus (and keep re-applying it), check that cameras, stage and light haven't moved since the freeze, re-aim view |
| `config/` | cell settings, inspection views, reference frames of the frozen cell (`cell_ref/`) |
| `results/` | anomaly-model evaluations (`results/anomaly/`), device bench |
| `docs/PLAN.md` | the plan written during the event (see its status note: not everything in it was built) |

## Arm tools
| | |
|---|---|
| `arm_test.sh` | identify joints, calibrate, verify, teleop |
| `teleop.sh` | verify + teleop (`tools/invert.json` lists joints to reverse in software; it is empty: the leader's wrist_flex direction is flipped in the servo, `tools/flip_dir.py`) |
| `camdash.sh` / `tools/camdash.py` | live view of the 3 cameras + snapshots, localhost only (`ssh -N -L 8080:localhost:8080 intel`) |
| `tools/v4lctl.py` | list/set camera controls: `python tools/v4lctl.py /dev/video0 auto_exposure=3` |

## Hardware (station laptop)
- Leader arm `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00`, follower `...5AE7045345-if00`
- Cameras (found by name, `tools/find_cams.py`): top = RealSense D415, inspection = "Webcam Audio", wrist = "USB2.0_CAM1"

## Repo rules
- Enforced by git hooks + CI ([CONTRIBUTING.md](CONTRIBUTING.md)). Once per clone: `git config core.hooksPath .githooks`.
- One program at a time on the arms. Before returning the laptop: [RETURN_CHECKLIST.md](RETURN_CHECKLIST.md).
