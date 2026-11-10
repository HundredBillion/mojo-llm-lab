# Longer whole-model decode: timing passes, numerical qualification fails

The Qwen3.5-2B candidate runs the complete model's fixed-token decode loop about **1.97–2.43% faster** in four independent confirmation processes. Both CUDA/CUDA controls differ less: 0.10% and 0.70%. This passes the timing decision recorded before confirmation. **It does not qualify the candidate for use:** a separate full-vocabulary comparison fails the preset numerical tolerance, despite matching every checked top token.

No production defaults changed. Prompt-processing, sampled generation, complete PDF latency, sustained temperatures and energy savings remain unproved. The 2× target has not been achieved.

## Protocol and timing

Same pinned Qwen3.5-2B-Q4_K_M model, llama.cpp b12110, 128-thread Mojo gate/up kernel, full GPU offload, Q4_0 KV, flash attention, two CPU threads with polling disabled, 20% CPU quota and verified blocking CUDA waits. Each process loads one model with two independent contexts, keeps the implementation fixed per context, primes two tokens, then runs 128 fixed decode tokens. Two warmups precede one ABBA cycle. Timed host replacement and capture counts are zero: the previously captured graphs execute the selected kernels.

The longer repeated synthetic fixture preserves the earlier fixture's first 42 token IDs. Longer runs cover different positions and must not be pooled with the [negative 16-token comparison](context-crossover.md).

| Stage | CUDA / Mojo latency ratio |
| --- | ---: |
| Exploratory, context 1 | 1.015369 |
| Exploratory, context 0 | 1.025612 |
| Exploratory, context 0 | 1.023007 |
| Exploratory, context 1 | 1.022208 |
| Confirmation, context 0 | 1.024323 |
| Confirmation, context 1 | 1.021135 |
| Confirmation, context 1 | 1.021288 |
| Confirmation, context 0 | 1.019656 |

Exploratory control ratios are 1.001735 and 1.003297; confirmation controls are 1.001042 and 1.007018. The [preregistered rule](TSPs/11-07-2026-long-resident-confirmation.md), committed in `d2728c4` before confirmation measurements, requires every candidate ratio above one and the smallest candidate log-ratio minus the largest absolute control log-ratio to exceed 0.005. The observed margin is **0.0124718593**. This diagnostic is not a confidence interval and cannot rule out all correlated or systematic error.

Recompute from the [raw evidence](../results/a2000/long-resident/):

```sh
python3 tools/verify_long_confirmation.py results/a2000/long-resident
```

The verifier checks artifact identities, settings, context ordering, capture/replacement evidence, fixed-prefix final top tokens and the decision rule. Passing it establishes the timing rule only; numerical evidence below independently rejects qualification.

## Full-output check

`resident-logits` is a separately compiled validation binary. It reads the full 248,320-entry vocabulary after the final prefill output, both priming tokens and every decode token: 131 observed positions. It compares three candidate trials and two repeated CUDA trials against the first CUDA reference. These instrumented timings are explicitly rejected by the performance summarizer.

Both context assignments produce identical numerical results:

- Candidate: 97,589,760 compared values; 86,336,859 exceed `1e-4 + 1e-5 * abs(CUDA reference)`; maximum absolute difference 1.6430788.
- Repeated CUDA: 65,059,840 values, all bit-identical to their reference.
- Candidate and CUDA top tokens match at every observed position. This is insufficient to establish equal output distributions or sampled behavior.

A separate diagnostic retains per-position differences. Observed positions 0–14 match exactly; position 15 first diverges, with maximum absolute difference 0.383988202. The largest difference occurs at position 93. Position zero is the final eight-token prefill output, not the first input token. Tolerances have not been relaxed.

## Isolating the division discrepancy

The generated Mojo PTX uses `div.rn.f32` for SiLU, whereas the pinned [CUDA build configuration](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-cuda/CMakeLists.txt) enables `-use_fast_math`. Both paths use an approximate exponential; attributing the difference to a precise CUDA exponential would be incorrect.

A diagnostic changed only the PTX division instruction to `div.approx.ftz.f32`. **Every candidate and repeated-CUDA logit then matched exactly**: 97,589,760 candidate and 65,059,840 repeated-control comparisons, no top-token mismatches. The edited PTX hash and single-instruction provenance are retained with its numerical-only results. This isolates that instruction as sufficient to eliminate the observed discrepancy on this fixture; it does not prove universal equivalence.

`kernels/q4_k_q8_1_gate_order_2b_fastdiv.mojo` expresses the change in source as a separate candidate. Its guarded Mojo build passed (CPU/GPU peaks 66°C/56°C). The compiled candidate then passed full-logit validation in **both context assignments**, again with exact equality for every compared value. Its PTX SHA-256 is `8e130a5a0a4e4355acbd0988bd31f627b09fdfcda31eedd7ce13fb05e52e9e3a`.

Fresh performance confirmation is still required before attributing any speed gain to this candidate. The numerical runs' instrumented timings are not performance evidence, and the original candidate's timing win cannot be transferred to the corrected candidate. The [new confirmation protocol](TSPs/11-07-2026-fast-division-confirmation.md) was committed before its measurements.

## Conditions and limits

Runs used a user-authorized bounded fan session at normal maximum level 7 with a hardware watchdog and automatic restoration. Sprite was briefly paused with an independent resume watchdog, then resumed between processes. All AC, CPU, GPU and platform temperature guards remained active. Some runs reached GPU 80°C, only 2°C below the abort threshold; these short runs do not establish sustained thermal safety.

The numerical binary does not overwrite the performance worker. The original performance worker hash stayed unchanged throughout exploration and confirmation. The controller gained an inactive-by-default numerical flag during confirmation; performance commands and workload stayed unchanged. Both the earlier negative evidence and these new numerical failures are retained.

## Corrected-candidate performance confirmation, attempt r2 (2026-11-07): incomplete

The first performance batch for the source-built fast-division candidate stopped after three of six processes. Stage 03 (Mojo context 1) failed its power/temperature preflight on a transient CPU 90.0 °C reading, the abort threshold. Completed ratios: control 0.99896574, Mojo context 0 1.02220388, Mojo context 1 1.01023753. These are directionally consistent with the original candidate but **do not satisfy, or fail, the preregistered rule**, which needs all six processes. They are not a qualification of the corrected candidate. Stage 01 also peaked at GPU 81 °C, 1 °C below the abort threshold.

Evidence, receipts and the exact helper copies: [`results/a2000/long-resident/fastdiv-confirmation-r2/`](../results/a2000/long-resident/fastdiv-confirmation-r2/). A further attempt must be a new, uniquely named, complete six-process batch. Before it, consider a longer cool-down between stages (the current 8 s gap allowed a CPU spike from 82 °C to 90 °C). That would be a protocol change and must be committed before measuring.

## Attempt r3 (2026-11-07): incomplete, GPU guard

With the [pre-committed cool-down amendment](TSPs/11-07-2026-fast-division-confirmation-r3.md), both stages started cool (5.2 s and 8.1 s waits). The control passed (ratio 1.0008548347150767, GPU peak 81 °C). Stage 01 (Mojo context 0) was **aborted mid-run** when the GPU reached the 82 °C guard. Per-run GPU peaks across r2 and r3 were 78–82 °C. This fixed-token protocol has no reliable thermal headroom on this laptop at fan level 7, and further identical attempts are likely to abort the same way. Evidence: [`results/a2000/long-resident/fastdiv-confirmation-r3/`](../results/a2000/long-resident/fastdiv-confirmation-r3/). The corrected candidate remains unconfirmed.

## Attempts r4 and r5 (2026-11-07): passed at a 1200 MHz GPU clock cap

[r4](../results/a2000/long-resident/fastdiv-confirmation-r4/) never measured: the GPU does not support power limiting. [r5](../results/a2000/long-resident/fastdiv-confirmation-r5/) locked the GPU graphics clock to ≤ 1200 MHz for all six processes ([amendment](TSPs/11-07-2026-fast-division-confirmation-r5.md), committed before measurement) and completed. Candidate ratios: 1.02649667, 1.02339584, 1.01561332, 1.02473248. Control ratios: 0.99683031, 1.00195822. Log-margin 0.0123179672 ≥ 0.005: **the source-built fast-division candidate passes the fixed-token decode confirmation, at the capped clock.** Combined with its earlier exact full-logit validation, this is the first corrected-candidate whole-model decode result, about 1.6–2.6 % lower latency. It does not cover default clocks, prompt processing, sampled output, PDF latency or the 2× target. GPU peaks were 79–80 °C and CPU peaks reached 89 °C.

## Broader numerical and generated-output coverage (2026-11-07): passed

A separate numerical-only worker prefilled four synthetic prompts: three using a long statement-extraction system prompt, up to 6772 tokens, and one non-financial prose prompt. It then generated 128 tokens free-running, four runs greedy and two seeded-sampled. Full logits matched CUDA **exactly** at every position in all six runs: 96.1 M candidate and 64.1 M repeated-CUDA values per run, maximum difference 0. Generated outputs were identical. See [`results/a2000/logit-coverage/`](../results/a2000/logit-coverage/) and the [protocol](TSPs/11-07-2026-logit-coverage.md). Long prefills required fan level 7 and GPU clock locks to stay under the 82 °C guard. Thermal stops are retained.
