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
| `camdash.sh` / `tools/camdash.py` | live camera dashboard, see below |
| `tools/v4lctl.py` | list/set camera controls: `python tools/v4lctl.py /dev/video0 auto_exposure=3` |

## Camera dashboard (everyone on the team)
Live view of all 3 cameras + snapshot button.

**1. Make sure it runs** (on the Intel laptop): `bash camdash.sh status` (also `start` / `stop` / `restart`)

**2. Open it from your own laptop** (needs your SSH access to `intel`, ask Marius):
```
ssh -N -L 8080:localhost:8080 intel
```
Leave that window open, then go to http://localhost:8080. Works the same in PowerShell, macOS and Linux.
Port 8080 busy on your laptop? Use `-L 8081:localhost:8080` and open http://localhost:8081.

**On the Intel laptop screen itself:** just open http://localhost:8080 (no lag, no Wi-Fi in between).

Notes:
- Each camera shows `cam X fps · you Y fps · Z ms` (camera rate, frames reaching you, latency). Junction Wi-Fi adds lag.
- **Close the tab before recording** with lerobot / Physical AI Studio: cameras are released ~3 s after the last viewer leaves.
- Snapshots are saved to `snapshots/<camera>/` on the Intel laptop.
- The server only listens on localhost, so it is not exposed to the whole Junction network.

## Docs
- [docs/CRISTIANA.md](docs/CRISTIANA.md): Romanian beginner guide (concepts, setup, plan)
- [RETURN_CHECKLIST.md](RETURN_CHECKLIST.md): before returning the laptop

## Rules
- **Repo rules are enforced** (git hooks + CI): see [CONTRIBUTING.md](CONTRIBUTING.md). Once per clone: `git config core.hooksPath .githooks`.
- Third-party software and licences: [THIRD_PARTY.md](THIRD_PARTY.md).
- One program at a time on the arms; announce on chat first.
- No reboot / system packages without announcing.
