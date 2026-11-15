# INT8 weight compression of the ACT policies (Sat 14 Nov, laptop time 20:00–20:45 PST)

Tool: NNCF 3.4.0 `compress_weights` (weight-only, per-channel), OpenVINO 2026.3 for building.
Evaluated with physicalai's `InferenceModel` (Studio venv, OpenVINO 2026.4.1), the same loader and
preprocessing the cell runtime uses. Studio's exports were read only; every compressed model lives in
`results/int8/<model>[-orig]/<variant>/` next to a copy of its `manifest.json`. Raw numbers: `int8.json`.

Drift = |INT8 action − FP action| in degrees (Studio units for the gripper), over the first 10 steps of
each predicted chunk, on real frames from each model's own training dataset, both run on CPU.
For scale: FP itself is 1–3.4° from the recorded demo action on the first step.

## 1. Full INT8 (all 78 weight layers, INT8_ASYM) — not safe

| Model | bin size FP → INT8 | Mean drift ° (worst joint) | Max drift ° |
|---|---|---|---|
| act-pick-v6-h100 | 138.1 → 35.8 MB | 35.3 (shoulder_lift) | 152.6 |
| act-place-good-v5 | 138.1 → 35.8 MB | 37.6 (shoulder_lift) | 133.4 |
| act-place-bad-v6 | 138.1 → 35.8 MB | 26.9 (shoulder_lift) | 105.2 |

100 frames per model. Building from the untouched `act.orig.xml` instead of the deployed (camera-warp +
gripper-map) export gives the same picture (pick, 20 frames: mean 47.2°, max 153.5° vs 43.6° / 152.6°).

Latency, pick model, end to end (50 calls, median / p90 ms) — partial, the run hit its time limit:

| Device | FP | INT8 |
|---|---|---|
| NPU | 16.38 / 17.12 | 13.04 / 13.49 |
| GPU | 7.92 / 9.36 | 7.33 / 9.25 |
| CPU | 82.62 / 89.07 | 61.62 / 69.15 |

(place-good: NPU 12.8 → 13.15 ms, GPU 8.61 → 9.15 ms; place-good CPU and place-bad were not reached.)

## 2. Where the error comes from (pick model, 20 frames, CPU)

| Variant | Layers in INT8 | bin MB | Mean drift ° (worst) | Max drift ° |
|---|---|---|---|---|
| all layers (as above) | 78 / 78 | 35.8 | 43.6 (shoulder_lift) | 152.6 |
| skip Gather | 78 / 78 | – | identical to all layers (no Gather weights get compressed) | |
| **Conv only (skip MatMul + Gather)** | 41 / 78 | 103.8 | **0.093 (elbow_flex)** | **0.61** |
| INT8_SYM, all layers | 78 / 78 | 35.8 | did not load in OV 2026.4.1 (`m_element_type.is_static()` check) | |

The model has 42 Convolution and 49 MatMul ops. Quantizing the ResNet backbone convolutions costs
under 0.7°, about the same as running FP on the NPU instead of the CPU (max 0.41°, `npu_drift.json`).
The damage comes from the transformer / action-head MatMuls (37 of the 78 compressed layers).

Conv-only latency, pick model, end to end (50 calls, median / p90 ms). This run was taken with a load
average of ~17 from other jobs, so absolute numbers are higher than in section 1; compare within the
table only. The run hit its 300 s limit before conv-only INT8 on CPU.

| Device | FP | Conv-only INT8 |
|---|---|---|
| NPU | 24.40 / 28.25 | 16.50 / 17.43 |
| GPU | 11.26 / 20.29 | 9.93 / 15.75 |
| CPU | 117.21 / 130.31 | not reached |

## 3. FP on NPU vs CPU (pick, 50 frames)

Deployed export: NPU vs CPU mean 0.02–0.06°, max 0.41° (shoulder_lift); GPU vs CPU max 0.25°.
No NaN, no saturation. The NPU plugin rejects `INFERENCE_PRECISION_HINT=f32` (config parse error).

## Verdict

- Full INT8 as built: **do not put on the arm.** Mean errors of 27–38° on the shoulder are not a
  rounding issue, they are a different trajectory.
- Conv-only INT8 is numerically safe (≤0.61° max on these frames) but saves only 25 % of the weights
  (138 → 104 MB), and it was not run on the arm.
- Not tried: data-aware INT8 (activation-calibrated PTQ with `nncf.quantize`), mixed precision on the
  MatMuls, INT4.
