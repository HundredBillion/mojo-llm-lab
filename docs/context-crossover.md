# Context crossover: no demonstrated whole-model gain

The planned six-run batch completed under unchanged AC/thermal guards. It does not establish a Mojo whole-model speedup. All runs used the same Qwen3.5-2B weights, worker/observer/PTX/wait-helper artifacts, verified blocking waits, 20% CPU allowance, two priming tokens and 16 fixed timed tokens per trial. Each process loaded one model and two contexts, warmed both, then measured one ABBA cycle. Assignment stayed fixed for each context's lifetime.

The schedule was recorded before measurement: CUDA control, candidate contexts 1, 0, 0, 1, CUDA control. No run was dropped or retried in this batch. Sprite was paused for approximately 22–29 seconds per run with an independent 80-second resume watchdog and 75-second experiment deadline; it resumed for eight seconds between runs. Chromium was absent. Pause receipts show no process left stopped.

| Run | Mojo context | CUDA latency / Mojo latency |
| --- | ---: | ---: |
| Candidate 1 |1 |0.96703× |
| Candidate 2 |0 |1.02599× |
| Candidate 3 |0 |0.99841× |
| Candidate 4 |1 |0.96891× |

Above 1 means lower Mojo latency. The four-ratio geometric mean is 0.98979×, an exploratory descriptive statistic rather than proof of a regression or equivalence. Both context 1 candidate runs were slower; context 0 was slightly faster once and approximately equal once. CUDA/CUDA controls also differed, with arm 0/arm 1 latency ratios 0.95951 and 0.90379. These data do not support a consistent implementation advantage.

Every timed candidate trial reported zero host replacement/capture calls after priming, while earlier cumulative counters confirmed candidate capture. These counters do not count graph replays. Fixed-prefix top token 4952 matched across every trial, a narrow check rather than full output/logit equivalence. Sampled peaks across the batch were CPU 77°C/GPU 81°C/platform 60.05°C; this is short-run evidence, not sustained thermal safety. GPU clocks were not locked.

## Implementation and verification

`--mojo-arm 0|1` selects the candidate context and defaults to 1 for compatibility. Worker handshake and trial rows must agree with the controller; CUDA/candidate counter validation follows the selected context. The summarizer emits `cuda_over_mojo_latency_ratios` in the correct direction for either assignment and no candidate ratio for a control. It rejects workload mismatches and partial assignment metadata; completely legacy records retain their historical arm 1 assignment.

The worker compiled under CPU/thermal guards and 79 CPU tests passed. Regression tests catch reversed candidate assignment, contradictory or missing metadata, invalid workload timing and the partial-metadata case found during focused review. The implementation review was scoped to crossover behavior, not full-PR clearance.

[Raw evidence, pause receipts and recomputed results](../results/a2000/context-crossover/comparison.json) retain all six runs. Reproduce with the existing resident command, `--continuous --prime-tokens 2 --cuda-wait blocking --mojo-arm 0` or 1, and a fresh output directory for each run. Keep model, artifact hashes, CPU allowance and workload matched; do not combine runs merely because their labels look similar.

## Decision

More whole-model trials of this unchanged 2B candidate are not the best next step. Directly qualify the isolated 2048-input/6144-output fused operation against pinned CUDA, preserving correctness and same-implementation controls. The earlier 1.22–1.26× operation result used the 4B shape and must not be transferred to 2B. Use that narrower diagnostic to decide what kernel change to try, then return to whole-model and sampled-output qualification. The PR's whole-model acceleration requirement remains unmet.
