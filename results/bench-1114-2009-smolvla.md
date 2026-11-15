# Device benchmark 2026-11-14T20:14:42

OpenVINO 2026.4.1-22982-e213a147257-releases/2026/4 on ? / Intel(R) Arc(TM) B390 GPU (iGPU) / ?. Batch 1, static shapes, random inputs, LATENCY hint; throughput with the THROUGHPUT hint. Compile = cold (no cache).

| model | weights | device | precision | compile s | first ms | median ms | p95 ms | inf/s (tput) | package W | iGPU W | NPU % | mJ/inf above idle | busy alongside |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SmolVLA smolvla-place-both-v5 | bf16 98%, f32 2% | GPU | f16 | 13.6 | 93.4 | 78.65 | 81.07 | 10.9 | 39.6 | 21.4 | 0 | 1796.88 | – |

**What it means for one brick:**
- SmolVLA smolvla-place-both-v5: one call returns a 50-step chunk (output [1,50,6], 1.7 s of motion at 30 Hz); 78.7 ms median on GPU = 236% of one 33 ms control step.

Numbers are measured on the station laptop by `tools/bench_devices.py`; power from `tools/power_log.py` (RAPL).
