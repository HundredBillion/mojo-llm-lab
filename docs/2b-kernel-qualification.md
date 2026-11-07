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

The 64-thread variant assigns two Q4 blocks per projection lane and reduces two warps. The 256-thread variant assigns 128 threads each to gate and up and preserves the original four-warp projection reduction order. Source review found no indexing or launch-selection defects, but neither variant has live numerical or performance qualification yet.

All 82 CPU tests passed. The 64-thread Mojo and native harness builds completed; its Mojo compile peaked at CPU 85°C/GPU 70°C. The subsequent 256-thread Mojo compilation was stopped by the unchanged guard at sampled CPU 92°C/GPU 70°C/platform 65.05°C, about 115 seconds into the stage, despite its 10% CPU scope allowance. The batch stopped there; the 256-thread native harness was not built and neither new layout was benchmarked. The partial PTX is not treated as a qualified build. The guard samples temperature and cannot guarantee zero overshoot between samples.

After cleanup, there was no GPU compute process or owned benchmark/build container; CPU had fallen to 75°C and GPU to 62°C. The post-stop process snapshot cannot establish which process caused the spike. No sustained safety or causal attribution is claimed. Logs and stage receipts retain the failed build.

Next: resume only after a cold window, investigate the compile-period CPU spike, qualify both geometries under the same numerical checks, and return to whole-model evidence. Do not enable either experimental geometry in the application or the resident integration based on these results.
