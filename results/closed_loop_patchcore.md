# Closed-loop results (pick -> inspect -> place, no human touch)

Made by `tools/closed_loop_report.py` from the run folders below and the power log. Only runs with inspection on (no --no-inspection / --force-verdict / --only).

## Runs

| run | cycles written / planned | policy device | anomaly model / device | pick / good / bad model | source |
|---|---|---|---|---|---|
| 20261114-195345 | 2/3 | GPU (openvino, async) | patchcore_views_b2g_384 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-195345/cycles.jsonl` |
| 20261114-195521 | 3/3 | GPU (openvino, async) | patchcore_views_b2g_384 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-195521/cycles.jsonl` |
| 20261114-195725 | 3/3 | GPU (openvino, async) | patchcore_views_b2g_384 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-195725/cycles.jsonl` |
| 20261114-200200 | 3/3 | GPU (openvino, async) | patchcore_views_b2g_384 / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-200200/cycles.jsonl` |
| 20261114-200539 | 2/3 | GPU (openvino, async) | patchcore_views_b2g_384_mask / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-200539/cycles.jsonl` |
| 20261114-202121 | 1/3 | NPU (openvino, async) | patchcore_views_b2g_384_mask / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-202121/cycles.jsonl` |
| 20261114-202337 | 1/3 | GPU (openvino, async) | patchcore_views_b2g_384_mask / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-202337/cycles.jsonl` |
| 20261114-202403 | 3/3 | NPU (openvino, async) | patchcore_views_b2g_384_mask / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-202403/cycles.jsonl` |
| 20261114-202926 | 1/1 | NPU (openvino, async) | patchcore_views_b2g_384_mask / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-202926/cycles.jsonl` |
| 20261114-203554 | 1/1 | NPU (openvino, async) | patchcore_views_b2g_384_mask / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-203554/cycles.jsonl` |
| 20261114-204723 | 1/1 | NPU (openvino, async) | patchcore_views_b2g_384_mask / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-204723/cycles.jsonl` |
| 20261114-204839 | 1/1 | NPU (openvino, async) | patchcore_views_b2g_384_mask / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-204839/cycles.jsonl` |
| 20261114-205016 | 1/1 | NPU (openvino, async) | patchcore_views_b2g_384_mask / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-205016/cycles.jsonl` |
| 20261114-205215 | 1/1 | NPU (openvino, async) | patchcore_views_b2g_384_mask / GPU | act-pick-v6-h100 / act-place-good-v5 / act-place-bad-v6 | `data/cell_runs/20261114-205215/cycles.jsonl` |

Skipped 2 run folders that are not closed-loop runs or wrote no cycle (e.g. 20261114-204008 (--only place_good), 20261114-211840 (no cycle written)).

## Summary

- Cycles attempted: **24**, sorted without help: **15** (62 %)
- Outcomes: sorted: 15, aborted: pick failed: 7, aborted: arm not at HOME at cycle start: 2
- Cycle time, sorted bricks (n=15): mean 60.5 s, p50 49.72 s, p95 108.95 s; pick p50 13.54 s, place p50 11.44 s, inspection p50 6141.3 ms (anomaly inference p50 1115.1 ms per frame of 3 views)
- Verdicts on sorted bricks: good 5, defect 10; uncertain 3 (go to the defect bowl), re-inspected 3; bowls: good 5, defect 10
- Verdict accuracy: no hand labels yet (fill `results/truth.csv` with run,cycle,truth)
- Energy per sorted brick (`results/power-1114-1353.csv`, 15 bricks, median): whole laptop 2133.4 J = 0.5926 Wh, CPU package 1321.5 J; mean draw 43.5 W; above idle (18.57 W, quietest 60 s window of the power log): 1223.3 J
- Thermals during cycles: package max 89.0 °C, min headroom to TjMax 11.0 °C; NPU busy 0 %, iGPU clock 1905 MHz (medians of cycle means)

## Cycles

| run | # | outcome | s | verdict | score (low/high) | looks | bowl | truth | stages (name:outcome:run s) |
|---|---|---|---|---|---|---|---|---|---|
| 20261114-195345 | 0 | sorted | 32.95 | good | -0.22 (1.00/2.00) | 1 | good | – | pick:done:13.54 place_good:done:11.44 |
| 20261114-195345 | 1 | aborted: arm not at HOME at cycle start | 0.0 | – | – | 0 | – | – |  |
| 20261114-195521 | 0 | sorted | 33.14 | defect | 3.49 (1.00/2.00) | 1 | defect | – | pick:done:12.87 place_bad:done:12.47 |
| 20261114-195521 | 1 | sorted | 37.83 | defect | 2.71 (1.00/2.00) | 1 | defect | – | pick:done:15.67 place_bad:done:9.87 |
| 20261114-195521 | 2 | aborted: arm not at HOME at cycle start | 6.48 | – | – | 0 | – | – |  |
| 20261114-195725 | 0 | sorted | 36.15 | defect | 2.67 (1.00/2.00) | 1 | defect | – | pick:done:12.57 place_bad:done:10.04 |
| 20261114-195725 | 1 | sorted | 49.59 | good | -0.65 (1.00/2.00) | 1 | good | – | pick:done:11.21 place_good:done:24.81 |
| 20261114-195725 | 2 | aborted: pick failed | 96.54 | – | – | 0 | – | – | pick:timeout:45.04 pick:failed:40.04 |
| 20261114-200200 | 0 | sorted | 60.73 | defect | 2.62 (1.00/2.00) | 1 | defect | – | pick:done:11.98 place_bad:done:36.61 |
| 20261114-200200 | 1 | sorted | 37.24 | good | 0.31 (1.00/2.00) | 1 | good | – | pick:done:13.07 place_good:done:11.17 |
| 20261114-200200 | 2 | sorted | 49.72 | defect (uncertain) | 1.68 (1.00/2.00) | 2 | defect | – | pick:done:20.57 place_bad:done:9.84 |
| 20261114-200539 | 0 | sorted | 114.01 | defect (uncertain) | 1.38 (1.00/1.56) | 2 | defect | – | pick:failed:30.84 pick:done:19.31 place_bad:failed:5.23 place_bad:done:12.45 |
| 20261114-200539 | 1 | aborted: pick failed | 43.82 | – | – | 0 | – | – | pick:failed:23.24 |
| 20261114-202121 | 0 | aborted: pick failed | 100.99 | – | – | 0 | – | – | pick:timeout:45.01 pick:timeout:45.01 |
| 20261114-202337 | 0 | aborted: pick failed | 3.15 | – | – | 0 | – | – | pick:stopped:0.07 |
| 20261114-202403 | 0 | sorted | 36.01 | good | 0.24 (1.00/1.56) | 1 | good | – | pick:done:14.32 place_good:done:12.92 |
| 20261114-202403 | 1 | sorted | 106.78 | defect (uncertain) | 1.52 (1.00/1.56) | 2 | defect | – | pick:timeout:45.01 pick:done:10.37 place_bad:done:24.58 |
| 20261114-202403 | 2 | sorted | 93.72 | defect | 2.05 (1.00/1.56) | 1 | defect | – | pick:timeout:45.01 pick:done:19.31 place_bad:done:10.48 |
| 20261114-202926 | 0 | sorted | 90.55 | defect | 5.30 (1.00/1.56) | 1 | defect | – | pick:done:12.02 place_bad:timeout:45.01 place_bad:done:13.64 |
| 20261114-203554 | 0 | sorted | 69.61 | defect | 5.09 (1.00/1.56) | 1 | defect | – | pick:done:28.7 place_bad:done:10.22 |
| 20261114-204723 | 0 | aborted: pick failed | 27.69 | – | – | 0 | – | – | pick:failed:8.64 |
| 20261114-204839 | 0 | aborted: pick failed | 10.19 | – | – | 0 | – | – | pick:failed:0.07 pick:failed:0.07 |
| 20261114-205016 | 0 | aborted: pick failed | 10.21 | – | – | 0 | – | – | pick:failed:0.08 pick:failed:0.07 |
| 20261114-205215 | 0 | sorted | 59.4 | good | 0.05 (1.00/1.56) | 1 | good | – | pick:done:17.09 place_good:done:11.25 |
