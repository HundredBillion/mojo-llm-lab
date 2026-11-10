# Numerical and generated-output coverage of the fast-division candidate (2026-11-07): ALL SIX RUNS PASSED

Protocol: [`docs/TSPs/11-07-2026-logit-coverage.md`](../../../docs/TSPs/11-07-2026-logit-coverage.md), committed in `e6decb7` before any measurement, with amendments committed before attempts 2–4. Candidate PTX `8e130a5a…`. Worker `resident-coverage` SHA-256 `2280b9ee652090819b967d18244313e1a4f37972b8ef4b4cf73c92f92f1335ca`. Prompts and their SHA-256 manifest are in [`experiments/coverage-prompts/`](../../../experiments/coverage-prompts/).

| Run | Prompt | Tokens | Policy | Candidate ctx | Attempt | Result |
| --- | --- | ---: | --- | ---: | --- | --- |
| 00 | statement-short | 1852 | greedy | 1 | [2](attempt2/) | passed, EOG at 81 |
| 01 | statement-medium | 3104 | greedy | 0 | [2](attempt2/) | passed, no EOG in 128 |
| 02 | statement-long | 6772 | greedy | 1 | [3](attempt3/) | passed, no EOG in 128 |
| 03 | prose | 566 | greedy | 0 | [3](attempt3/) | passed, EOG at 97 |
| 04 | statement-short | 1852 | sampled, seed 20261107 | 1 | [4](attempt4/) | passed, EOG at 83 |
| 05 | statement-long | 6772 | sampled, seed 20261107 | 0 | [4](attempt4/) | passed, no EOG in 128 |

Each passing run compared 96,099,840 candidate logits (3 trials × 129 positions × 248,320) and 64,066,560 repeated-CUDA logits. Each had **0 tolerance violations, 0 top-token mismatches and maximum absolute difference exactly 0**, and all six trials of each run produced the identical 128-token generated sequence and text. Candidate graph capture was non-zero in every run, and CUDA contexts performed no replacement.

Stopped attempts, retained and not counted: [attempt 1](attempt1/) (GPU 82 °C at default clocks), attempt 2 run 02 (GPU 82 °C at 1200 MHz), attempt 3 run 04 (CPU 90 °C launch spike). No completed run was repeated.

**Scope:** for this candidate, model, these synthetic prompts (up to 6772 prompt tokens plus 128 generated) and this laptop, the candidate's full logits and its greedy and seeded-sampled generated outputs are identical to CUDA. This is not universal equivalence. It says nothing about prompt-processing speed (prefill does not use the candidate), PDF latency or extraction quality: for example, `statement-medium` produced a category outside the prompt's allowed list without the server's grammar constraint, identically on both paths. Runs used GPU clock locks (1200 or 900 MHz), which do not change the instruction stream.

**Prompt provenance:** the three `statement-*` prompts in these recorded runs embedded a longer extraction system prompt from a private application, which is not published. Their recorded `prompt_sha256` values and token counts (1852, 3104 and 6772) identify those prompts. `tools/generate_coverage_prompts.py` now writes a short generic system prompt instead, so regenerated `statement-*` prompts are shorter and will not match the recorded hashes. `prose` is unchanged.
