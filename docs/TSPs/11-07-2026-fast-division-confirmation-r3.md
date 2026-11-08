# Amendment r3: inter-stage cool-down for the fast-division confirmation

Amends [`11-07-2026-fast-division-confirmation.md`](11-07-2026-fast-division-confirmation.md). Committed before any r3 measurement.

Attempt r2 stopped after three of six processes: in the fixed 8 s gap after stage 02 (CPU peak 82 °C), the CPU rose to 90 °C and stage 03 failed the runner's power/temperature preflight. That batch is retained as incomplete and is not resumed or filled in.

The only change for r3 replaces the fixed 8 s inter-stage gap with a cool-down wait before **every** stage, including the first. The batch samples the runner's own `run_benchmark.Sensors` every 0.25 s and starts the stage only after CPU < 70 °C and GPU < 68 °C have held for 5 s continuously. The batch stops (no stage started) if any sample fails the existing safety guard (AC, CPU ≥ 90 °C, GPU ≥ 82 °C, platform ≥ 75 °C) or no cool window appears within 180 s. Each stage's wait result is recorded in the batch JSON.

Everything else is unchanged:
- the six-process order control, Mojo 0, Mojo 1, Mojo 1, Mojo 0, control;
- worker, observer, model, wait library and source-built PTX `8e130a5a…`;
- 128 tokens, two priming tokens, one ABBA cycle, 20 % CPU quota and blocking CUDA wait;
- the runner and its thermal guards, the bounded level-7 fan window and the pause watchdogs;
- the decision rule and the separate numerical gate.

r3 is a complete new batch with new output names. A stop on safety or harness failure is kept as an incomplete result and is not retried on the basis of its partial ratios. Helper copies: [`results/a2000/long-resident/fastdiv-confirmation-r3/helpers/`](../../results/a2000/long-resident/fastdiv-confirmation-r3/helpers/).
