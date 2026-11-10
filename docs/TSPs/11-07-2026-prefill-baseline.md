# Measure prompt processing (prefill) separately

Committed before any prefill measurement.

## Why

The plan asks for prompt processing to be measured separately from decode. The qualified Mojo candidate replaces only the single-token Q4_K gate/up matrix-vector launch. In all six [coverage runs](../../results/a2000/logit-coverage/), each candidate trial recorded only 55–104 replacements while prefilling 566–6772 prompt tokens in 32-token batches. If the candidate ran in prefill, a 6772-token prompt would produce about 212 batches × 24 layers ≈ 5,000. The candidate therefore **does not run during prefill and cannot change prefill speed**. This step measures the CUDA prefill baseline and where its time goes, so any later prefill kernel has a target.

## Measurements

Each run is one guarded process of pinned `llama-bench` (b12110), with these settings matching the reference server configuration: full offload, Q4_0 K/V, flash attention, 2 threads, polling off, 20 % CPU quota (1000/5000 µs) and verified blocking CUDA waits. No Mojo PTX or observer is loaded. `-n 0` measures prompt processing only. llama-bench performs its own warmup.

| Run | Prompt tokens | batch / ubatch | Repetitions | Purpose |
| --- | ---: | --- | ---: | --- |
| 1 | 512 | 32 / 32 | 3 | reference-setting throughput |
| 2 | 2048 | 32 / 32 | 3 | reference-setting throughput |
| 3 | 6144 | 32 / 32 | 3 | reference-setting throughput, long context |
| 4 | 2048 | 128 / 128 | 3 | batch-size sensitivity |
| 5 | 2048 | 512 / 512 | 3 | batch-size sensitivity |
| 6 | 2048 | 32 / 32 | 1 | Nsight Systems kernel attribution (throughput discarded) |

Results are descriptive: mean and per-repetition tokens/second. There is no pass/fail decision. Runs 4–5 only measure the setting. Server settings are not changed, and VRAM, quality and latency effects of a larger ubatch in the real server are not established.

## Conditions and safety

Default GPU clocks are used so that throughput is not capped. The batch runs inside a bounded root fan session at level 7 with a 30 s hardware watchdog and automatic restoration. Before each run, the batch idles 20 s and then requires CPU < 70 °C and GPU < 68 °C held for 5 s (at most 180 s). The runner itself applies the cold start and the absolute guards: AC; abort at CPU ≥ 90 °C, GPU ≥ 82 °C or platform ≥ 75 °C; 240 s deadline. On a guard stop, the batch stops and keeps the incomplete result. A clock-locked retry would be a separate, documented amendment whose throughput is reported as capped.

## Outputs

`results/a2000/prefill-baseline/` contains each run's summary, sensors and logs; the Nsight report hash and a kernel aggregation by `tools/summarize_cuda_profile.py`; and a report of the prefill operation shares at the reference server settings, compared with the decode shares in `docs/qwen35-2b-profile.md`.
