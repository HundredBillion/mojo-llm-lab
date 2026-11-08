# Broaden numerical and generated-output coverage of the fast-division candidate

Committed before any coverage measurement.

## Why

The source-built fast-division candidate (PTX `8e130a5a0a4e4355acbd0988bd31f627b09fdfcda31eedd7ce13fb05e52e9e3a`) matched CUDA exactly on one fixture: an 8-token prefill of a repeated synthetic invoice sentence, then 130 teacher-forced positions. That fixture is short, repetitive and never lets the model choose its own tokens. The earlier rounded-division failure showed that top tokens can match while other logits drift, so coverage must compare full logits, not only outputs.

## What is measured

A separate numerical-only worker, `integration/resident_coverage.cpp` (binary `build/integration/resident-coverage`), is used. The performance worker `resident-decode`, the `resident-logits` validator and their hashes are unchanged. Its timings are never performance evidence.

For each run, the worker:

1. tokenizes a prompt file (special tokens parsed, no added BOS) and prefills it in 32-token batches into two independent contexts sized to the prompt plus 128 generated tokens (rounded up to 256, maximum 8192). The KV cache uses Q4_0 and flash attention, as before.
2. generates 128 tokens **from the model's own output** (free-running, not teacher-forced) with a fixed policy:
   - `greedy`: argmax of the context's own logits (lowest index on ties);
   - `sampled`: a fresh llama.cpp sampler chain per trial (top-k 40, top-p 0.95, temperature 0.8, fixed seed), so identical logits imply an identical token stream.
3. compares the full vocabulary (248,320 logits) at all 129 output positions, from the final prefill output through the last generated token. The tolerance is unchanged: `1e-4 + 1e-5 * abs(CUDA reference)`. Top-token mismatches are counted separately.
4. runs six trials with fixed context assignment, control context `c = 1 - m`, in the order `c, m, m, c, m, c`. This gives three candidate trials and three CUDA trials; the first CUDA trial is the reference, so two repeated-CUDA comparisons serve as controls.
5. reports every trial's generated token IDs and the decoded text up to the first end-of-generation token.

## Prompts

`tools/generate_coverage_prompts.py` deterministically writes the prompt files and a SHA-256 manifest to `experiments/coverage-prompts/`. All content is synthetic; no user document is read or modified. Prompts use the Qwen chat format with thinking disabled (empty think block), approximating the the application server's `enable_thinking=false` template.

| Prompt | Content | Approx. tokens (exact counts recorded by the worker) |
| --- | --- | ---: |
| `statement-short` | full the application bank-transaction extraction system prompt (~9.5 KB) + the existing one-transaction synthetic statement | ~2.5k |
| `statement-medium` | same system prompt, 2 pages, 40 synthetic transactions | ~3.5k |
| `statement-long` | same system prompt, 6 pages, 150 synthetic transactions | ~6–7k |
| `prose` | non-financial: summarize synthetic project meeting notes | ~0.7k |

## Schedule

Six runs, each a separate guarded process:

| Run | Prompt | Policy | Candidate context |
| --- | --- | --- | ---: |
| 1 | statement-short | greedy | 1 |
| 2 | statement-medium | greedy | 0 |
| 3 | statement-long | greedy | 1 |
| 4 | prose | greedy | 0 |
| 5 | statement-short | sampled, seed 20261107 | 1 |
| 6 | statement-long | sampled, seed 20261107 | 0 |

## Pass rule

A run passes only if:

- the report is complete (expected position and value counts);
- there are zero candidate and zero repeated-CUDA tolerance violations and top-token mismatches;
- all six trials generated the identical token sequence;
- candidate graph capture is non-zero and the CUDA context performed no replacement.

The coverage claim passes only if all six runs pass. Any failure is retained and reported. Tolerances are not relaxed, and runs are not repeated to obtain a pass. A run stopped by a safety guard is reported as incomplete.

## Safety

Each run uses the existing absolute guards: AC power; abort at CPU ≥ 90 °C, GPU ≥ 82 °C or platform ≥ 75 °C; a cold start (CPU < 70 °C, GPU < 68 °C held for 5 s) before every process; the CPU quota; and blocking CUDA waits. Timing is not measured, so desktop processes are not paused. The worker build and CPU tests run under the existing 10 % CPU-quota guarded build. If a run trips a thermal guard, the batch stops and a fan session is considered as a separate, documented change.

## Scope

A pass extends numerical agreement to longer contexts, the application-style extraction prompts and free-running greedy and sampled generation, for this candidate, model, prompts and laptop. It does not establish universal equivalence, prompt-processing speed (prefill is not replaced by the candidate), complete PDF latency or extraction quality.

## Amendment (attempt 2)

Committed after attempt 1 and before attempt 2. [Attempt 1](../../results/a2000/logit-coverage/attempt1/) stopped on the GPU 82 °C guard during the first CUDA trial of run 1. The long prompt prefill loads the GPU more heavily than decode-only fixtures.

Attempt 2 reruns the complete six-run schedule from the start (new output names, `local-coverage-r2-*`) inside a bounded root session. That session sets the fans to level 7 with a 30 s hardware watchdog, locks the GPU graphics clock to 210–1200 MHz and enables persistence mode temporarily. Everything is restored on every exit path, as in the [r5 confirmation](11-07-2026-fast-division-confirmation-r5.md). Before each run, the batch requires an active window with at least 300 s remaining and valid fan readings. After each run, it requires a recorded maximum SM clock of at most 1215 MHz.

Clock frequency does not change kernel selection or the instruction stream, so the numerical pass rule is unchanged. If the 1200 s window ends before all six runs, the remaining runs continue in a new, separately recorded window. Completed runs are not repeated.

## Amendment (attempt 3)

Committed after attempt 2 and before attempt 3. [Attempt 2](../../results/a2000/logit-coverage/attempt2/) passed runs 1–2. These are retained and not repeated. Run 3 (`statement-long`, 6772 prompt tokens) reached the GPU 82 °C guard during its first CUDA trial at ≤ 1200 MHz.

Runs 3–6 (`local-coverage-r3-*`) use these changes, which leave the numerics unaffected:

- the GPU clock lock is lowered to 210–900 MHz (verified per run: max recorded SM clock ≤ 915 MHz);
- the controller's `--trial-cooldown` waits for the existing cold window (CPU < 70 °C, GPU < 68 °C for 5 s, at most 60 s) before **each** of the six trials, not only before the process. The overall worker deadline grows from 420 s to 1200 s to allow for these waits.

Everything else is unchanged, including the pass rule, the abort thresholds and the fan session. If the window runs short, the remaining runs continue in a new, recorded window.

## Amendment (attempt 4)

Committed after attempt 3 and before attempt 4. [Attempt 3](../../results/a2000/logit-coverage/attempt3/) passed runs 3–4, which are retained. Run 5 failed its process preflight on a transient CPU 90 °C reading when it launched immediately after run 4, matching the fast-division r2 stop. Before **each** remaining run (5 and 6, `local-coverage-r4-*`), the batch now idles 20 s and then waits for CPU < 70 °C and GPU < 68 °C held for 5 s (at most 180 s), stopping on any unsafe sample. This is the [fast-division r3](11-07-2026-fast-division-confirmation-r3.md) cool-down. All other attempt-3 settings are unchanged, including the ≤ 900 MHz clock lock and the per-trial cool-down.
