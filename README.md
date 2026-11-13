# CheckurAI

Junction 2026, Intel Physical AI Challenge: SO-101 arm picks Duplo bricks, shows them to an
inspection camera, Anomalib decides good/defective, the arm drops them in the matching bowl.

## Hardware (Intel laptop, `ssh intel`)
- Leader arm:   `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00`
- Follower arm: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00`
- Cameras (found by name, see `tools/find_cams.py`): top = RealSense D415, inspection = "Webcam Audio", wrist = "USB2.0_CAM1"
- Env: `conda activate intel_dev_env`

## Tools
| | |
|---|---|
| `arm_test.sh` | identify joints, calibrate, verify, teleop |
| `teleop.sh` | verify + teleop with reversed joints (`tools/invert.json`) |
| `tools/camdash.py` | live camera dashboard, see below |
| `tools/v4lctl.py` | list/set camera controls: `python tools/v4lctl.py /dev/video0 auto_exposure=3` |
| `anomaly/capture.py` | save inspection-camera images to `data/anomaly/{good,defect}` |
| `anomaly/train.py` | train PaDiM/PatchCore on the Intel GPU, export to OpenVINO |
| `anomaly/detect.py` | OpenVINO detector (CPU/GPU/NPU), live mode and `--bench` |
| `bench/power.py` | package power and temperature logger for the judges |

Setup (Studio, Modal credits, anomaly workflow, scoring): [docs/SETUP.md](docs/SETUP.md).

## Camera dashboard
On the Intel laptop: `python tools/camdash.py` (already running in background, log in `logs/camdash.log`).
From your own laptop: `ssh -N -L 8080:localhost:8080 intel`, then open http://localhost:8080.
Close the tab before recording with lerobot / Physical AI Studio (cameras are released ~3 s after the last viewer).
Snapshots are saved to `snapshots/<camera>/`.

## Rules
- One program at a time on the arms; announce on chat first.
- No reboot / system packages without announcing.
