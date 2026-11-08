# Coverage attempt 2 (2026-11-07): runs 1–2 passed, run 3 stopped by the GPU thermal guard

The session ran with fans at level 7 and the GPU clock locked to ≤ 1200 MHz (max SM clock recorded: 1200 MHz in each run).

| Run | Prompt / policy | Prompt tokens | Result |
| --- | --- | ---: | --- |
| 00 | statement-short / greedy, candidate ctx 1 | 1852 | **passed**: 96,099,840 candidate and 64,066,560 repeated-CUDA logits compared, 0 violations, 0 top mismatches, max abs difference 0; identical 128-token output in all 6 trials (EOG at token 81); 93 candidate captures; peaks CPU/GPU 79/81 °C |
| 01 | statement-medium / greedy, candidate ctx 0 | 3104 | **passed**: same counts, max abs difference 0, identical output (no EOG within 128); 108 captures; peaks 83/81 °C |
| 02 | statement-long / greedy, candidate ctx 1 | 6772 | **stopped**: GPU reached 82 °C during the first CUDA trial; no comparison |

The fan/clock session restored with no errors (fan auto, watchdog 0, `llm-fan.service` active, persistence mode Disabled, clocks reset). Runs 3–6 continue in attempt 3 under the [amendment](../../../../docs/TSPs/11-07-2026-logit-coverage.md#amendment-attempt-3).
