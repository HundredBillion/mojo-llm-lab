# Coverage attempt 3 (2026-11-07): runs 3–4 passed, run 5 stopped at preflight

The session ran with fans at level 7, the GPU clock locked to ≤ 900 MHz (max SM clock recorded: 900 MHz) and a cool-down before every trial.

| Run | Prompt / policy | Prompt tokens | Result |
| --- | --- | ---: | --- |
| 02 | statement-long / greedy, candidate ctx 1 | 6772 | **passed**: 96,099,840 candidate and 64,066,560 repeated-CUDA logits, 0 violations, 0 top mismatches, max abs difference 0; identical output in all 6 trials (no EOG within 128); peaks CPU/GPU 80/74 °C |
| 03 | prose / greedy, candidate ctx 0 | 566 | **passed**: same counts, max abs difference 0, identical output (EOG at 97); peaks 83/66 °C |
| 04 | statement-short / sampled, seed 20261107, candidate ctx 1 | — | **stopped at preflight**: CPU read 90.0 °C (GPU 59 °C idle) when the process launched immediately after run 03. No trial ran. |

The session restored with no errors. Runs 5–6 continue in attempt 4 under the [amendment](../../../../docs/TSPs/11-07-2026-logit-coverage.md#amendment-attempt-4).
