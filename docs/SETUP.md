# CheckurAI setup

Challenge guide: [JUNCTION2026.md on the `junction2026` branch of physical-ai-studio](https://github.com/open-edge-platform/physical-ai-studio/blob/junction2026/JUNCTION2026.md).
Organizers asked everyone to train and run inference from that branch. On the laptop
`~/physical-ai-studio` is already on `junction2026`.

## Laptop access

See the Wi-Fi + SSH note from Marius (`ssh intel`, key-only). One key per person in
`~/.ssh/authorized_keys`, everyone shares the `intel` user. `conda activate intel_dev_env`
has torch 2.10 (XPU), OpenVINO 2026.3 (CPU/GPU/NPU), LeRobot 0.4.4 and Anomalib 2.6.0.

## Physical AI Studio

```bash
# terminal 1, backend on :7860
cd ~/physical-ai-studio/application/backend && uv run --extra xpu physicalai-studio serve
# terminal 2, UI on :3000 (node comes from nvm)
source ~/.nvm/nvm.sh && cd ~/physical-ai-studio/application/ui && npm run start
```

From your own laptop: `ssh -N -L 3000:localhost:3000 -L 7860:localhost:7860 intel`, then open
http://localhost:3000. Stop `teleop.sh` and close camdash before Studio connects to the arms/cameras.

## Modal (policy training on cloud GPUs)

Every participant gets $100. Credits belong to a workspace and each workspace redeems the code once,
so each of us redeems in our own workspace (code and form: Modal's Junction Notion page).

1. Sign up and add a card **over a phone hotspot**, not the venue Wi-Fi (fraud checks block it).
2. Settings → Usage & Billing: set the spend limit to $0, so the card is never charged.
3. Redeem the Junction code, check Settings → Usage shows the $100.
4. Add your token on the laptop as your own profile (prints a link, approve it in your browser):

   ```bash
   uvx --python 3.13 --from 'modal[api-proxy-support]>=1.6.3,<2' modal token new --profile <name> --expires-in 3d
   uvx --python 3.13 --from 'modal[api-proxy-support]>=1.6.3,<2' modal profile list
   ```

5. When someone's credits run out: `modal profile activate <other>`, restart the backend, then in
   Studio Settings → Training Targets → Modal Provider → **Deploy and connect** again.

GPU choice: A10G for ACT/SmolVLA, A100-80GB or H100 only for Pi0.5 (plain "A100" is the 40 GB card).
Jobs keep billing even if Studio is closed; before sleeping run `modal app list` and stop what is running.
Backup if Modal runs out: Studio can also train on any Ubuntu GPU host over SSH (Verda credits).

## Anomaly detection (Anomalib, runs on the laptop, no Modal)

```bash
python anomaly/capture.py good --count 60 --roi x,y,w,h   # good bricks only, vary pose and light
python anomaly/capture.py defect --count 20                # optional, used as test set
python anomaly/train.py --model padim                      # or patchcore --int8, trains on XPU
python anomaly/detect.py models/anomaly/padim --bench      # CPU vs GPU vs NPU latency/power table
python anomaly/detect.py models/anomaly/padim --cam inspection --device NPU
```

`Detector` in `anomaly/detect.py` is the piece the closed loop imports.

## Evidence for judges

`bench/power.py` samples package power (RAPL, via `sudo -n`) and CPU temperatures.
`python bench/power.py --idle 10` for a baseline, `python bench/power.py -- <command>` around any run.
`detect.py --bench` writes results to `logs/bench/`.

## Scoring (two versions exist, ask organizers which one is used)

| GitHub guide | pts | Kickoff PDF | pts |
|---|---|---|---|
| Task success (accuracy, cycle time, robustness) | 25 | End-to-end closed loop | 25 |
| Intel tech (NPU/iGPU, OpenVINO, Robotics AI Suite) | 25 | Anomalib defect detection | 20 |
| Dataset & pipeline quality | 20 | VLA & Physical AI Studio | 20 |
| System efficiency (latency, thermals, power) | 20 | OpenVINO & Core Ultra optimization | 20 |
| Creativity | 10 | Robotic execution & reliability | 10 |
| | | Innovation & demo | 5 |

## Extra Intel repos (creativity)

- `open-edge-platform/geti-instant-learn`: few-shot localization (SAM2-style) on OpenVINO/NPU, to find
  and crop the brick before Anomalib, more robust to shifted objects and clutter.
- `open-edge-platform/edge-ai-suites` → `robotics-ai-suite/pipelines/pi05-rtc-ov` (Pi0.5 + real-time
  chunking on OpenVINO) and `act-sample` (ACT on OpenVINO). Built for ALOHA, need SO-101 adaptation.
- Qualia (partner): flags bad teleop episodes before training.
