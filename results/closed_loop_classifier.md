# Closed-loop results (pick -> inspect -> place, no human touch)

Made by `tools/closed_loop_report.py` from the run folders below and the power log. Only runs with inspection on (no --no-inspection / --force-verdict / --only).

## Runs

| run | cycles written / planned | policy device | anomaly model / device | pick / good / bad model | source |
|---|---|---|---|---|---|
| 20261114-214312 | 1/1 | NPU (openvino, async) | resnet18_synth_v1 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-214312/cycles.jsonl` |
| 20261114-214410 | 1/1 | NPU (openvino, async) | resnet18_synth_v1 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-214410/cycles.jsonl` |
| 20261114-214717 | 1/1 | NPU (openvino, async) | resnet18_synth_v1 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-214717/cycles.jsonl` |
| 20261114-215341 | 1/1 | NPU (openvino, async) | resnet18_synth_v1 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-215341/cycles.jsonl` |
| 20261114-215610 | 1/1 | NPU (openvino, async) | resnet18_synth_v1 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-215610/cycles.jsonl` |
| 20261114-215746 | 1/1 | NPU (openvino, async) | resnet18_synth_v1 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-215746/cycles.jsonl` |
| 20261114-220111 | 1/1 | NPU (openvino, async) | resnet18_synth_v1 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-220111/cycles.jsonl` |
| 20261114-220459 | 1/1 | NPU (openvino, async) | resnet18_synth_v1 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-220459/cycles.jsonl` |
| 20261114-220809 | 1/1 | NPU (openvino, async) | resnet18_synth_v1 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-220809/cycles.jsonl` |
| 20261114-221034 | 1/1 | NPU (openvino, async) | resnet18_synth_v1 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-221034/cycles.jsonl` |
| 20261114-221155 | 1/1 | NPU (openvino, async) | resnet18_synth_v1 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-221155/cycles.jsonl` |
| 20261114-221313 | 1/1 | NPU (openvino, async) | resnet18_synth_v1 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-221313/cycles.jsonl` |

Skipped 1 run folders that are not closed-loop runs or wrote no cycle (e.g. 20261114-214902 (no cycle written)).

## Summary

- Cycles attempted: **12**, sorted without help: **9** (75 %)
- Outcomes: sorted: 9, aborted: pick failed: 3
- Cycle time, sorted bricks (n=9): mean 54.68 s, p50 41.06 s, p95 112.98 s; pick p50 20.82 s, place p50 14.74 s, inspection p50 544.3 ms (anomaly inference p50 6.38 ms per frame of 3 views)
- Verdicts on sorted bricks: good 4, defect 5; uncertain 0 (go to the defect bowl), re-inspected 2; bowls: good 4, defect 5
- Verdict accuracy vs hand labels (8 bricks, `results/truth.csv`): **62 %**

| true \ bowl | good bowl | defect bowl |
|---|---|---|
| good | 1 | 0 |
| defect | 3 | 4 |

- Energy per sorted brick (`results/power-1114-1353.csv`, 9 bricks, median): whole laptop 1588.3 J = 0.4412 Wh, CPU package 840.1 J; mean draw 36.2 W; above idle (29.24 W, quietest 60 s window of the power log): 360.9 J
- Thermals during cycles: package max 91.0 °C, min headroom to TjMax 9.0 °C; NPU busy 0.5 %, iGPU clock 2255 MHz (medians of cycle means)

## Cycles

| run | # | outcome | s | verdict | score (low/high) | looks | bowl | truth | stages (name:outcome:run s) |
|---|---|---|---|---|---|---|---|---|---|
| 20261114-214312 | 0 | sorted | 27.57 | good | 0.03 (0.42/0.71) | 1 | good | good | pick:done:12.38 place_good:done:12.81 |
| 20261114-214410 | 0 | aborted: pick failed | 60.64 | – | – | 0 | – | – | pick:timeout:45.01 pick:no_motion:10.01 |
| 20261114-214717 | 0 | sorted | 41.06 | defect | 0.98 (0.42/0.71) | 1 | defect | defect | place_bad:done:39.49 |
| 20261114-215341 | 0 | sorted | 110.8 | defect | 0.91 (0.42/0.71) | 1 | defect | – | pick:done:38.38 place_bad:timeout:45.02 place_bad:done:14.74 |
| 20261114-215610 | 0 | sorted | 38.12 | defect | 1.00 (0.42/0.71) | 1 | defect | defect | pick:done:11.98 place_bad:done:23.75 |
| 20261114-215746 | 0 | sorted | 40.27 | defect | 1.00 (0.42/0.71) | 1 | defect | defect | pick:done:27.48 place_bad:done:10.51 |
| 20261114-220111 | 0 | aborted: pick failed | 40.83 | – | – | 0 | – | – | pick:failed:26.55 |
| 20261114-220459 | 0 | aborted: pick failed | 72.76 | – | – | 0 | – | – | pick:failed:27.58 pick:failed:33.64 |
| 20261114-220809 | 0 | sorted | 29.78 | defect | 1.00 (0.42/0.71) | 1 | defect | defect | pick:done:12.21 place_bad:done:11.93 |
| 20261114-221034 | 0 | sorted | 45.84 | good | 0.14 (0.42/0.71) | 1 | good | defect | pick:done:14.16 place_good:done:27.64 |
| 20261114-221155 | 0 | sorted | 44.22 | good | 0.32 (0.42/0.71) | 2 | good | defect | pick:done:28.48 place_good:done:11.72 |
| 20261114-221313 | 0 | sorted | 114.44 | good | 0.26 (0.42/0.71) | 2 | good | defect | pick:done:41.02 place_good:timeout:45.01 place_good:done:16.74 |
