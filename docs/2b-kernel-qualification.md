# 2B fused-operation qualification

The existing 128-thread Mojo gate/up kernel passed two isolated tests at the Qwen3.5-2B dimensions: 2048 inputs and 6144 distinct output rows. Both paths include identical Q8_1 activation quantization, use the same synthetic Q4_K weights, and pass numerical tolerance and output red-zone checks. Each run has eight ABBA rounds and a preceding same-CUDA-graph control.

| Run | CUDA median µs | Mojo median µs | Median ratio | Paired geometric mean | CUDA control ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 |109.488 |92.880 |1.1788× |1.1394× |0.99983× |
| 2 |115.663 |99.904 |1.1577× |1.1378× |1.00058× |

Ratios above one favor Mojo. All 16 paired rounds favored Mojo. Timing shifts within the runs make the paired statistic more conservative than the ratio of overall medians; neither is a confidence interval or evidence of stationary timings. This is promising isolated-operation evidence, not whole-model acceleration. The previous context crossover still shows no consistent whole-model gain.

Mojo used 34 registers versus CUDA's 56, no local memory in either kernel, and reported a theoretical maximum of 12 versus 9 blocks per SM. This is occupancy capacity, not measured occupancy. Both used two graph nodes including quantization. Recorded SM clocks were 1207 MHz and memory clocks 5501 MHz. Sampled peaks across both runs were CPU 62°C/GPU 64°C/platform 58.05°C. These short tests do not establish sustained cooling performance.

Sprite was paused for about 18–19 seconds per experiment with an independent 80-second resume watchdog; pause receipts confirm it resumed. Chromium was absent. Existing thermal, AC and duration limits were unchanged. No attempts were discarded.

## Reproduction

Generate fixtures with `python3 tools/generate_q4_matvec_fixtures.py --gate-up --random-quants --rows 6144 --input-width 2048 --output build/q4-fixtures-2b`. Build the existing PTX with `tools/build_q4.sh gate-order-2b`, then `python3 tools/build_q4_comparison.py --shape 2b --gate-up --threads 128`. Run `python3 tools/run_q4_cuda_comparison.py --variant gate-order-2b --fixtures build/q4-fixtures-2b --output results/local-2b-fresh` under the thermal guards. Use the pinned versions and an idle GPU; the runner refuses active compute. The private desktop-pause controller is not needed for correctness but the measured desktop conditions should be reproduced for timing comparisons.

[Raw results and fixture hashes](../results/a2000/2b-kernel-qualification/comparison.json) retain both controls, all paired samples, sensor traces and pause receipts. The existing 4B evidence remains separate.

## Experimental geometry and thermal stop

The 64-thread variant assigns two Q4 blocks per projection lane and reduces two warps. The 256-thread variant assigns 128 threads each to gate and up and preserves the original four-warp projection reduction order. Source review found no indexing or launch-selection defects, but neither variant had live numerical or performance qualification at the end of this initial batch.

All 82 CPU tests passed. The 64-thread Mojo and native harness builds completed; its Mojo compile peaked at CPU 85°C/GPU 70°C. The subsequent 256-thread Mojo compilation was stopped by the unchanged guard at sampled CPU 92°C/GPU 70°C/platform 65.05°C, about 115 seconds into the stage, despite its 10% CPU scope allowance. The batch stopped there; the 256-thread native harness was not built and neither new layout was benchmarked. The partial PTX is not treated as a qualified build. The guard samples temperature and cannot guarantee zero overshoot between samples.

After cleanup, there was no GPU compute process or owned benchmark/build container; CPU had fallen to 75°C and GPU to 62°C. The post-stop process snapshot cannot establish which process caused the spike. No sustained safety or causal attribution is claimed. Logs and stage receipts retain the failed build.

The follow-up below resumes after cold gates and retains the original failure. Do not enable either experimental geometry in any application or the resident integration based on these results.

## Follow-up: 64 threads pass; 256 timing remains pending

The 64-thread variant passed a guarded control/comparison with the same full-row fixture and numerical tolerance. CUDA's median was 114.560 µs and Mojo's 98.432 µs (1.16385×); the eight paired ratios had a geometric mean of 1.14229× and all favored Mojo. The CUDA/CUDA control ratio was 1.03217×. This one run does not distinguish 64 threads from the existing 128-thread results (~1.138–1.139× paired). Mojo used 40 registers and no local memory. Sampled peaks were CPU 60°C/GPU 65°C.

The 256-thread rebuild completed under the same CPU and thermal limits, peaking at CPU 67°C/GPU 65°C. A half-second host recording showed the compiler near its 10% CPU allowance; the prior 92°C spike did not recur and remains unattributed. The build reused the same source and environment but was not a cold compiler-cache experiment.

The subsequent 256-thread benchmark completed its control (ratio 0.99943×, correctness and canaries passed), then stopped at the candidate preflight when CPU reached 94°C with GPU 62°C. Sprite was paused with its independent resume watchdog. No candidate timing was collected, and the failure is retained. A later three-minute observation without GPU work peaked at CPU 74°C/GPU 65°C; this does not explain the earlier spike.

Read-only inspection found the host's installed fan service filters specifically for its own managed model container; these benchmark containers do not match. Fan readback was automatic, approximately 2486/2341 RPM. This is a cooling-policy coverage gap, not proof that it caused either spike. A temporary maximum-normal-fan experiment is prepared but still requires administrator authentication; no persistent host setting has been changed.

A GGUF header inspection also identifies Q6_K token embeddings of dimensions [2048, 248320], plus Q6_K feed-forward down projections and Q5_K attention projections. Type enums were checked against the pinned GGML header. These are potential optimization targets, not timing attribution. The next whole-model diagnostic should profile the actual 2B model; the older 4B profile cannot establish this model's bottlenecks. The Q4_K replacement alone does not cover those operations.

The whole-model acceleration requirement remains unmet. Both primary user metrics—prompt processing and token generation—still need credible gains with preserved output quality; 2× remains an aspiration rather than a condition for reporting smaller verified gains.

## Completed 256-thread comparison under the fan window

After authentication, the three-minute normal-maximum-fan window enabled a complete 256-thread comparison. CUDA's median was 107.903 µs and Mojo's 109.568 µs (CUDA/Mojo 0.98480×); paired geometric mean was 0.98413×, with only two of eight rounds favoring Mojo. The CUDA/CUDA control ratio was 0.99519×. Numerical checks and canaries passed. Sampled peaks across control and comparison were CPU 57°C/GPU 52°C/platform 46.05°C.

The 256-thread kernel uses 29 registers versus the 128-thread kernel's previously observed 34, but its theoretical resident-block capacity is six rather than twelve. Lower register count did not translate into a speed improvement. One run does not prove a general regression, but it supplies no reason to replace the qualified 128-thread candidate. The changed cooling conditions prevent treating cross-run timing differences as a fan-only effect.

The fan helper restored automatic control, original watchdog 0 and the host fan service. The same window also produced the [current 2B model profile](qwen35-2b-profile.md), which now guides whole-model testing and further kernel selection. All earlier thermal failures remain retained.
