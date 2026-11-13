# CheckurAI: Intel "See It. Detect It. Act on It." plan

Decided on Fri 13 Nov 2026, ~23:30 Helsinki time. **The laptop clock is on PST (Helsinki minus 10 h)**, so timestamps in logs, datasets and git use PST.

This is the team's working plan. Every person and agent works from it. If a decision changes, change it here in the same commit.

## 1. Goal and how it's scored
The cell picks Duplo bricks from a shifted or cluttered pile, inspects each brick from every side (tiny marker dots in odd places included), and drops it in the good or defective bowl, fully autonomously. It runs on the Intel laptop using the Physical AI Studio → OpenVINO Physical AI runtime pipeline.

| Criterion | Weight | What earns it here |
|---|---|---|
| Task success | 30% | Multi-view inspection. A 3D grasp check with automatic re-grasp. Robustness to clutter, shifted bricks and moved cameras. Measured accuracy, cycle time and a confusion matrix. |
| Intel integration | 30% | Studio record → train (Modal) → export. `physicalai` runtime API, `SharedCamera`, `RerunCallback`. OpenVINO on CPU, iGPU and NPU, with a live device switch. NNCF INT8. |
| Dataset & pipeline | 20% | About 110 clean demos across 3 camera poses. Arms homed before every episode, every episode ended cleanly. A demos-vs-success curve. A self-improving data loop. |
| Efficiency | 20% | A device matrix. Pinning the policy to P-cores. Package power (RAPL), joules per brick, temperature, cycle time p95. |

**Hard rule from Intel:** the cell must run on the Intel station laptop and use Intel's Physical AI software. A cell built only on LeRobot, or run on our own laptops, doesn't count.

## 2. Decisions
| # | Topic | Decision |
|---|---|---|
| D1 | What "3D" means | A live 3D twin (Rerun) plus 3D decisions (supervisor checks). Depth is **not** fed into the policy. |
| D2 | Camera → robot calibration | Automatic, with a brick in the gripper: about 10 poses, brick centroid from RGB-D compared with FK, Kabsch fit. Reject if RMS > 15 mm. Re-runs in about 1 min. |
| D3 | Camera changes | Find cameras by serial or USB name, never by `/dev/videoN`. A "camera moved" detector. 3D re-calibrates automatically. **The policy is trained on 3 top-camera placements, with augmentation.** |
| D4 | Policy split | ACT `pick` (pick a brick and lift it), then a **scripted** inspection dance, then ACT `place_good` or ACT `place_bad`. SmolVLA with task strings is a stretch goal. |
| D5 | Inspection | Active multi-view sweep (V0–V6, §5) with a tight crop and its own model per view. Score = max over views of the normalised score. |
| D6 | Uncertain bricks | Re-present at a new angle once. If still uncertain, it goes to the **defective bowl** (fail-safe) and is logged as `uncertain→reject`. A third "review" spot is a stretch goal. |
| D7 | Done signal | Pose + 3D: gripper closed, arm near the lift pose and still for about 0.3 s, **and** depth shows a brick between the fingers. A timeout triggers a re-grasp (at most 2). |
| D8 | Run path | Studio is used to record, train and export. **Our orchestrator runs the exported models through the `physicalai` runtime Python API** (`RobotRuntime`, `PolicySource.reset()/warmup()`, `SharedCamera`). We don't drive Studio's websocket. |
| D9 | Training compute | **Modal**, through Studio (Settings → Training Targets → Modal Provider → Deploy and connect). Bogdan provides the credentials. The local Arc B390 (XPU) is only for the smoke test. There is no Intel cloud. |
| D10 | Efficiency | A full device matrix (policy × anomaly × {CPU, GPU, NPU} × {FP32, INT8 weights-only}). Ship the setting with the best task success, then the lowest J/brick. Devices can be switched live. |
| D11 | Arm calibration | The LeRobot calibration on the servo EEPROM is the master. **In Studio's SO-101 wizard, always pick "Skip verification"** (it reads the calibration from the board). Never run Studio's calibration step: it rewrites the homing offsets. |
| D12 | Safety | Soft stop: joint and velocity clamps, a workspace box checked against the twin, a watchdog that turns torque off, and the spacebar kill on the dashboard. **Hard stop: one person sits at the follower's 12 V switch during every autonomous run.** |
| D13 | Cut order (if behind on Sat 18:00) | Cut the data loop first, then the camera-bump demo, then SmolVLA. **Never cut:** ACT pick/place, multi-view Anomalib, the 3D grasp check, the twin with the heatmap, the device benchmarks. |
| D14 | Team | Marius and the Claude agents write the code. Teammates help with teleop, making defects and filming. |
| D15 | Bricks | We have fewer than 15. **Photograph every brick clean before marking anything.** Use removable defects so each brick can be both good and defective. |
| D16 | Conveyor | **Full 40 cm belt**: 2 × 200 mm modules, 100 mm wide, with an end chute into a ~12×12 cm pick tray. A teammate prints it overnight on the venue printers. **Drive:** the Atech 14-port board with a DC-motor module, controlled from the laptop. **Feed:** the belt runs until RealSense depth sees a new brick in the tray, then stops for the pick; it advances during the previous brick's placement. The pick policy also handles bricks on the table (judges may place bricks by hand). **Demos are recorded only after the belt is fixed in place, and never wait past Sat 09:30;** if the belt isn't ready by then, record without it. |

## 3. Station layout (tape it, photograph it, don't move it)
- **Table zones A–F** (pick area, about 25×20 cm), the **inspection spot**, the **good bowl** and the **defective bowl**, all taped and labelled.
- **Cameras:**

  | Camera | Role |
  |---|---|
  | `top` | RealSense D415 RGB+depth. 640×480 for the policy, at most 1280×720; 1080p gives black frames. |
  | `wrist` | Sonix, MJPG. |
  | `inspection` | Sunplus, MJPG. |

  Camera names must be identical in Studio, the exports and the runtime.
  - **3 taped mount positions for `top`** (P1/P2/P3), used for the demos (D3).
- **Inspection light:** a lamp or LED panel at a fixed position, with a phone flashlight as fallback. A white card behind the inspection pose. **Exposure and white balance are locked**, never auto.
- **Hard stop:** the follower's 12 V supply switch is within reach of the safety person.
- **Conveyor (D16):**
  - The belt runs across the table depth, about 15–20 cm left of the follower base, from the far edge (infeed) to the pick tray at the follower's front-left.
  - **The inspection webcam must move out of that strip tonight.** Record the dance poses only after it has moved.
  - Before fixing anything, tape the footprint. Check with the leader in teleop that the gripper reaches the whole tray, and check that the tray is in the RealSense view.
  - Plug the Atech board **directly into the laptop, not the hub**.
  - **Never commit the 3MF/STL files** (MakerWorld "Conveyor Generator", personal-use licence).

## 4. Architecture
```
            SharedCamera (iceoryx2), one publisher per camera
 top D415 RGB-D ─┬──────────────┬───────────────┐
 wrist ──────────┤              │               │
 inspection ─────┤              │               │
                 v              v               v
        ACT policies       Anomalib views     3D supervisor
        (OpenVINO, iGPU*)  (OpenVINO, NPU*)   (depth → base frame)
                 \              |               /
                  └──── orchestrator.py (state machine) ────┐
                        RobotRuntime(SO101) + safety        │
                                                            v
              Rerun: URDF twin + point cloud + brick boxes + defect heatmap
                     + device/latency/W/°C panel + counters        (* = default; matrix decides)
```
**State machine** (each arrow is a check; failures loop back):
```
FEED: belt on until depth sees a new brick in the tray (overlaps with the previous PLACE) | 15 s → line empty → table zones
IDLE ─[3D: brick in tray or on table? pick target]→ PICK (ACT pick, reset+warmup)
PICK ─[D7 done: closed+still+brick-in-gripper]→ INSPECT   | timeout/empty → RE-GRASP (≤2) → IDLE
INSPECT = scripted dance V2..V6 (+V0 captured pre-pick, V1 wrist close-up)
       → score s = max_v(s_v / thr_v):  s<0.8 GOOD | s>1.2 DEFECT | else RE-PRESENT once → else DEFECT(uncertain)
GOOD/DEFECT → PLACE (ACT place_good | place_bad) ─[3D: brick left gripper & is in bowl]→ LOG → IDLE
any state ─[safety violation / watchdog / spacebar]→ STOP (torque off)
```

## 5. Inspection (tiny dots in odd places)
| View | Camera | What it sees | Cost |
|---|---|---|---|
| V0 | top | top face, before the pick | free |
| V1 | wrist | close-up while grasped | free |
| V2–V5 | inspection | wrist roll 0/90/180/270°, so all 4 sides | about 0.3 s each |
| V6 | inspection | tilted up, so the underside and the hollow inside | about 0.4 s |

**Dance poses** are fixed joint targets in `configs/inspect_poses.json`, recorded once with the leader arm and replayed with ramped moves. Every brick is therefore seen from the same viewpoint, which is what lets PatchCore catch tiny dots.

**Crop:** the brick sits at a known spot in the frame at each pose. A colour/depth blob refines the box, and the crop is made at the **highest resolution the camera gives**. For tiny dots, tile the crop rather than shrinking it to 256 px.

**Models:**
- PatchCore (WideResNet50 layers 2+3, **coreset ≈ 0.1** so it fits on the NPU), one per view.
- Fallbacks: EfficientAD (lowest latency, NPU-friendly) or PaDiM.
- Thresholds are set per view on held-out good and defective images, then normalised to 1.0.

**Data protocol (fewer than 15 bricks):**
1. **Clean pass first:** every brick × 2–3 grasps × every view gives about 30–45 good images per view. Use **every colour**, so the model can't learn "colour = defect".
2. **Then add removable defects:**
   - Tiny marker dots (1–2 mm) in odd places: underside, inside, edges, between studs.
   - Larger scribbles.
   - Tape or stickers.
   - Covered or filled studs.
3. **Hold back one defect type** (for example dots on the underside), never used in training or for thresholds. It's the "new defect type" demo.

**OpenVINO notes:**
- Exports already include ImageNet normalisation, so feed RGB scaled to [0,1].
- **NPU:** static input shape [1,3,H,W]; set `CACHE_DIR`.
- **GPU:** set `INFERENCE_PRECISION_HINT=f32` if the anomaly maps drift from CPU.
- Anomalib trains on CPU (Lightning has no XPU accelerator).

**Twin:** the anomaly heatmap of the worst view is projected onto that face of the brick model in Rerun.

## 6. Policies and data
**Demos (about 110, about 2 h of arm time, two operators alternating):**
- `pick`: 60 = 3 top-camera placements (P1/P2/P3) × 20.
  - About half are from the **belt tray**: random landings and small piles made by running the belt for several bricks.
  - The other half are from table zones A–F, about 10 of them in clutter.
  - **Every episode ends cleanly at the lift pose.**
- `place_good`: 25 and `place_bad`: 25, each starting from the `HOME_INSPECT` pose with a brick in the gripper.
- **Before every episode, drive both arms home** (otherwise the start pose drifts). Move every joint a bit across the demos: a joint that never moves gets near-zero std and freezes the policy.
- Task strings and camera names must be **exactly** the same everywhere; a typo creates a new task.
- **2 bricks are never used in demos** (held-out test).

**Training (Studio → Modal):**
- ACT on pick-60, place-good-25 and place-bad-25.
- The **curve:** ACT pick on 10, 25 and 50 episodes, run in parallel on Modal.
- Choose checkpoints by **closed-loop success on the robot**, not validation loss.
- Execute whole action chunks; compare against temporal ensembling.
- Export: OpenVINO plus the runtime bundle. **Set `chunk_size` / `n_action_steps` explicitly at runtime**, because the exported metadata can say 1.
- **Stretch:** SmolVLA (pick + place with task strings), on Modal.

**Failure-driven top-up (Sat afternoon):** run 20 trials, log the failures with 3D context, then record 15–20 demos aimed at those failure states and retrain. Don't record "recovery" demos from states that also appear in good demos: that teaches the policy to let go.

## 7. 3D subsystem (`cell/perception3d/`)
- **Calibration** (`calib.py`, D2):
  - Hold a brick and move through about 10 joint poses covering the workspace.
  - At each pose, take the brick centroid from the aligned RGB-D image (colour mask + depth median).
  - Compute FK from the SO-101 URDF (`so101_new_calib.urdf`, shipped with the physicalai mujoco plugin) and add the gripper→brick offset.
  - Fit with Kabsch. Report RMS and the per-point error. Save `configs/T_cam_base.json` with a timestamp.
- **Point cloud:** the D415's depth-aligned-to-colour (`read_rgbd()`), cut to the workspace box, voxel ~3 mm, transformed into the base frame.
- **Checks** (`checks.py`):
  - Bricks on the table: clusters above the table plane, giving each brick's xyz and a clutter score.
  - Brick in the gripper: points inside the gripper box, using FK.
  - Brick left the gripper and is in the bowl.
  - Workspace safety box.
- **Camera-moved detector:** compare the current table plane and the edges against the reference.
  - Plane normal off by more than 2°, or offset by more than 1 cm: pause, re-calibrate automatically (about 1 min), resume.
  - Inspection and wrist cameras: compare against reference images (ORB/ECC). If moved, warn and re-capture the references.
- **Twin** (`twin.py`): Rerun with the URDF driven by the joint states, the point cloud, the brick boxes coloured by verdict, the gripper box, the heatmap, and the counters. The runtime's `RerunCallback` adds latency, CPU and GPU.

## 8. Runtime, devices, efficiency (`cell/bench/`)
- **Device matrix:** policy {CPU, GPU, NPU} × anomaly {CPU, GPU, NPU} × {FP32, INT8 weights-only (NNCF `compress_weights` INT8_ASYM)}.
  - Measured per cell: success %, cycle time mean and p95, control Hz, inference latency (`InferenceLatencyBenchmark`), package W (RAPL `/sys/class/powercap/intel-rapl:0/energy_uj`), J/brick, °C.
  - Full INT8 is tested but expected to hurt success. Choose by success.
- **CPU scheduling:**
  - The policy runs on the P-cores (`/sys/devices/cpu_core/cpus`); anomaly and 3D run on the E-cores (`cpu_atom`).
  - Count control steps over 40 ms with and without pinning.
  - Try the Robotics AI Suite core-pinning and frequency-locking scripts.
- **Compile cache** for every OpenVINO model. Warm up every policy at start, and `reset()` between bricks.
- Every number in the deck comes from a script in `cell/bench/`, writing `results/*.json`.

## 9. Hygiene (lost-time traps)
- **One program on the arms at a time:** Studio, the orchestrator, calibration and tests all compete for them. Say it in the chat.
- **Cameras:** one publisher each. Stop `camdash.py` before Studio or the runtime opens the cameras.
- **Stale iceoryx2 files** (`/tmp/iceoryx2/services/*`, `/dev/shm/iox2_*`) make connect hang; clean them only when nothing is running.
- **USB:** webcams always MJPG (USB 2 bandwidth). D415 at most 1280×720; give it about 15 warm-up frames. Lock exposure after every replug.
- **Servos:** ramp every scripted move; step commands trip the shoulder_lift overload latch. Turn gripper torque off between runs.
- **Never commit** datasets, models or credentials. **Back up datasets and exports off the laptop** (it may be wiped).

## 10. Timeline (Helsinki time)
| When | What |
|---|---|
| **Fri → 03:00** | Belt prints queued as early as possible (teammate). Follower recalibration (full base sweep), then save a new known-good config. Studio up; cameras added as `top`/`wrist`/`inspection`; the robot added with "Skip verification". Move the inspection webcam out of the belt strip. Tape zones, the belt footprint, cameras, lamp. **Smoke test:** 5 demos → ACT for about 2k steps on the XPU → OpenVINO export → `physicalai run` moves the arm. Record the dance poses. Modal target connected (Bogdan's credentials). Atech toolchain plus belt firmware flashed, motor spin test. |
| Sat 06:00–08:30 | Belt assembly, motor coupler, belt mounted and fixed in place; `belt.py` feed test. |
| **Sat 07:00–09:00** | **Clean anomaly capture** of every brick on the arm (the dance doesn't need the belt). Then make the defects and capture them. Calibrate 3D. |
| **Sat 09:00–12:00** | Demos with the belt fixed (start no later than 09:30, with or without the belt): pick 60 (P1/P2/P3, half from the tray), then place_good 25 and place_bad 25. Upload and start Modal training (main + curve runs). |
| Sat 09:00–12:00 (agents, in parallel) | 3D checks, twin, per-view anomaly models. |
| **Sat 12:00** | Intel hands-on session (5th floor). Ask about the NPU for ACT, xpu-smi, Robotics AI Suite pinning, and power measurement. |
| Sat 12:30–18:00 | Orchestrator integration. First autonomous runs. Failure-driven top-up demos and retrain. Per-view thresholds. |
| **Sat 18:00** | **Checkpoint:** is the full loop working? Apply the cut order (D13) if not. |
| Sat 18:00–24:00 | Device matrix, INT8, pinning, power. Camera-bump recovery. Data loop. SmolVLA if on track. |
| Sun 00:00–06:00 | Robustness runs, an evaluation over 30+ bricks for the numbers. Sleep in shifts. |
| **Sun 06:00** | **Code freeze.** Record the 2-min video, write the deck PDF and the README. |
| **Sun 09:30** | Submit FINAL (hard deadline 10:00): video, demo link, source link (repo made public), deck. |
| Sun 10:30– | Judging at the table. The whole team present. |

## 11. Workstreams (one agent each; files they own)
| WS | Owns | Deliverable |
|---|---|---|
| W1 Studio & data | Studio config, recording protocol, Modal training, exports | `exports/{pick,place_good,place_bad}` + the curve runs |
| W2 3D | `cell/perception3d/*`, `configs/T_cam_base.json` | calibration, checks, camera-moved detector, twin |
| W3 Inspection | `cell/inspect/*`, `configs/inspect_poses.json` | dance, capture tool, per-view models, thresholds, OpenVINO exports |
| W4 Orchestrator & safety | `cell/orchestrator.py`, `cell/safety.py`, `cell/devices.py` | state machine, camera discovery by serial, kill/watchdog |
| W5 Bench | `cell/bench/*`, `results/` | device matrix, power, pinning, evaluation and confusion-matrix scripts |
| W6 Story | `docs/`, the deck, the video | numbers slide, 2-min video, README |
| W7 Belt | `belt/` (Atech firmware `project.yaml`), `cell/belt.py`; the teammate does printing and assembly | the belt runs from the laptop; depth-gated feed; bricks/min |

## 12. What the judges see (rehearse it)
1. **Live run:** 8–10 mixed bricks (tiny dots in odd places, the held-out defect type), shifted and cluttered, running continuously with live counters.
2. **Big screen:** the Rerun twin with the defect heatmap on the brick, plus the device panel (where each model runs, latency, W, °C). Switch devices live.
3. **Camera bump:** knock the top camera, the cell detects it, re-calibrates and keeps sorting.
4. **Numbers slide:** accuracy and confusion matrix over 30+ bricks, cycle time mean and p95, the device matrix with J/brick, and the demos-vs-success curve. Every number comes from a script.

## 13. Risks
| Risk | Mitigation |
|---|---|
| Studio or runtime export path breaks | Smoke test tonight. Keep the exported bundle and its README command. |
| Modal credentials or quota | Fall back to ACT on the local XPU (about 50 min per model; it blocks the laptop). |
| A model doesn't compile on the NPU | Smaller coreset, EfficientAD or PaDiM, static shapes. Otherwise iGPU for anomaly, NPU for nothing, and say so honestly. |
| Policy poor on clutter | 3D picks the most isolated brick zone first. Failure-driven demos. Re-grasp. |
| Tiny dots missed | Higher-resolution tiles, better light, lower view threshold, the V6 underside view. Report recall per defect type honestly. |
| Belt late, or failing on demo day | Demos never wait past 09:30. The pick policy is also trained on table zones, so the cell works without the belt. A belt failure while running means the 15 s timeout, then table picks. |
| USB hub drops | Powered hub. Wrist camera on the main hub. `find_cams.py` after any re-cable. |
| The laptop gets wiped or returned | Push code often. Back up data and exports to the ThinkPad or cloud. |

## 14. After the hackathon: what to improve
- **Policy:**
  - A 3D-conditioned policy (point cloud or a calibrated multi-camera 3D representation) for real viewpoint invariance. The camera-pose augmentation we use now is only a patch.
  - A language-conditioned VLA that targets one specific brick ("the red 2×4 at the left"), with the target chosen by the 3D supervisor.
  - A learned stage/done classifier instead of pose heuristics. DAgger-style corrections with the leader arm during autonomous runs.
- **Inspection:**
  - Next-best-view: choose the next view from the current uncertainty instead of a fixed sweep.
  - Synthetic defects (procedural or diffusion) to set thresholds without damaging bricks.
  - Defect segmentation and size estimation in mm, using depth.
- **Sim and testing:** a MuJoCo SO-101 twin for regression tests and sim-to-real data; replay of logged runs.
- **Hardware performance:**
  - Accuracy-aware INT8 quantization (NNCF) and NPU deployment of the policy with a static-shape export.
  - A PREEMPT_RT kernel and isolated cores for deterministic control.
  - Pipelining the inspection of brick N while placing brick N-1.
- **Measurement:** wall power (not just RAPL), long-run thermals, and an MTBF/soak test.
- **Product:** a per-brick QC record (images, scores, verdict) for traceability. Multi-arm cells. Integration with a factory MES or dashboard.

## 15. Open items
- [ ] **Calibration status (from the calibration session, Fri ~23:16):**
  - **Leader:** recalibrated after the wrist_flex Phase flip.
  - **Follower: not recalibrated.** Its base was swept through only 172°, against 224° on the leader, so it sits about 27° off mid. Accepted for now.
  - **Known-good config:** `~/checkurai/config-good-1113-1316/` (copy on the ThinkPad: `junction2026/intel-config-good/`). Its `restore_config.py` has never been run.
  - **Rule:** any recalibration must happen **before** the real demos. Recalibrating afterwards makes the recorded dataset invalid.
  - **Teleop:** the "teleop v3" slew limit is LeRobot-side. Studio records with its own teleop, so check that it feels the same before recording.
- [ ] Before starting Studio: `pgrep -af "teleop|lerobot|camdash"`. Teleop v3 or camdash may still hold the ports and cameras. Ask Marius before killing anything.
- [ ] Modal credentials from Bogdan → Studio Training Targets → Modal.
- [ ] **Belt:**
  - Which port the motor uses: run `atech check` with the board plugged in, then fix `belt/project.yaml` and rebuild.
  - **USB alone does not power the motor rail.**
    - Get the **Power Jack module** (or USB-C PD) plus a supply from the hardware desk.
    - Ask the Atech people for the motor's rated voltage before going above about 6–9 V. Never go above 12 V.
    - The power module needs no `project.yaml` entry, and its port stays unused.
  - The motor is part of the module: shaft Ø3 mm, 8.8 mm long, about 10 mm above the board.
    - Use an edge port. The board sits on a printed cradle beside the belt's drive end, with a coupler from 3 mm to the TT double-D.
    - Generate the belt's 3MF with the TT-motor drive side.
  - **Speed check:** `dc_motor_speed 255` with no load, count turns in 10 s. Target belt speed is about 4–6 cm/s.
  - Ask Intel (Rajesh/Marvin) whether a conveyor fixture is OK and how the judges will place bricks.
- [ ] Lamp or LED panel from the Hardware Lab.
- [ ] Sunplus inspection-camera maximum resolution and fixed focus/exposure (check with `v4l2-ctl`, which isn't installed yet: `sudo apt install v4l-utils`).
- [ ] Revert the leader wrist_flex Phase flip before returning the laptop (see the RETURN_CHECKLIST).
