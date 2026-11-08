# A2000 kernel speed evidence and the model-quality gate

[Alternative-model follow-up](alternative-models.md): Qwen3.5-2B shows memory headroom; chat-format tests pass, but whole-model Mojo timing remains inconclusive.

The pinned `gate-order` Mojo implementation runs the tested fused Q4_K gate/up + SiLU operation faster than llama.cpp b12110's fused CUDA implementation on this RTX A2000 Laptop GPU. Three runs using identical artifacts and the broader synthetic fixture measured **1.2219×, 1.2206×, and 1.2583×** median operation-speed ratios. This is a bounded, reproducible operation result, not a claim that Mojo is universally faster than CUDA.

**The whole-model replacement is not qualified.** A deterministic invoice completion regresses even while all intermediate projection comparisons satisfy the existing tolerance. Do not enable this experimental replacement for production or PDF processing. No whole-model throughput gain, PDF latency gain, or cooling benefit is claimed.

## Operation evidence

| Run | CUDA median µs | Mojo median µs | CUDA / Mojo | Same-graph control difference |
| --- | ---: | ---: | ---: | ---: |
| Earlier qualification | 267.680 | 219.072 | 1.2219× | 0.13% |
| Earlier repeat | 270.464 | 221.584 | 1.2206× | 0.78% |
| Fresh PR qualification | 248.928 | 197.824 | 1.2583× | 0.12% |

Each run contains eight ABBA rounds, 32 samples total (16 per implementation), with 32 serialized graph repetitions per sample and two warmup pairs. All **24 paired rounds** favor Mojo; the smallest paired ratio is 1.1633×. Paired geometric-mean ratios are 1.2250×, 1.2368× and 1.2347×. Within-run rounds are correlated; these are not 24 independent experiments or a statistical guarantee about other workloads.

Both arms contain the same Q8_1 activation-quantization work and the fused gate/up operation. Both graphs have two kernel nodes. The input is 9728 distinct rows × 2560 columns, with varied factors and activation scales (seeds 739/740 and 971). Each run checks finite outputs, `absdiff <= 1e-5 + 1e-5*abs(CUDA)`, exact captured Q8 quantization bytes, and intact output canaries. The fresh run has maximum absolute difference 0.0000610351562 and 1925 non-bit-identical output values. Passing this contract does not imply bit identity or model-quality equivalence.

The fresh run's driver attributes report Mojo 38 registers/thread versus CUDA 56, zero local-memory bytes for both, and theoretical maximum resident blocks/SM of 12 versus 9. This is a plausible contributor to the difference, not a proven exclusive cause. Sampled SM/memory clocks were matched at 1207/5501 MHz. This was a repeatedly executed, cache-resident isolated operation; model execution accesses other tensors and is a separate workload.

Hardware: ThinkPad P1 Gen 4, i7-11850H, 64 GB RAM, RTX A2000 Laptop GA107/sm_86/4 GB, driver 615.78.08. Baseline: llama.cpp b12110, revision `cd959bae99ff4bed22bef2604f45241c09bbb8c8`. Candidate source is `kernels/q4_k_q8_1_gate_order.mojo`, compiled with Mojo 1.2.0/MAX 26.7.0; exact artifact/fixture hashes and pinned CUDA image are retained in each summary. Turbo, automatic fans and power policy were not changed. The fresh comparison peaked at CPU 60°C/GPU 67°C; its separate control peaked at 67°C/68°C.

[Fresh raw measurements](../results/a2000/q4-pr-proof/summary.json), [recomputed evidence](../results/a2000/q4-pr-proof/proof.json), and [previous method/results](q4-reduction-order.md) are retained. Recompute the numerical claim without a GPU:

```sh
python3 tools/verify_gate_evidence.py
```

The verifier rejects missing/failed qualification, broken canaries, incomplete rows, unmatched graph node counts, mislabeled controls, unmatched sampled clocks, a control beyond 5%, or differing artifacts/fixtures across runs. GPU reproduction uses the commands in [reduction-order.md](q4-reduction-order.md#reproduce); no model weights are needed for that operation benchmark.

## What integration established—and what it rejected

The explicit replacement mode supports only the pinned fusion ABI and exact Q4_K symbol, 2560-wide/9728-row layout, native strides, SiLU and absent bias/scale/routing/shared-expert features. Everything unsupported continues through CUDA. The module is prepared before capture in a single long-lived CUDA context, retained until process teardown, and launched on the original stream. Dynamic shared memory and launch attributes are not replaced. This research bridge does not support context destruction/recreation or a general multi-context service lifecycle.

A 16-token `llama-bench` smoke run completed with 72 replaced host launches, 36 captured launches, and CUDA graphs enabled. A separate feedback-validation run compared every eligible call, then copied the checked Mojo output into the model: 612 calls / 5,953,536 values passed. These results alone do not demonstrate generated-text equivalence.

The natural-language test requested one JSON object for a synthetic invoice (Oak Street Bakery, $127.50), using greedy decoding, seed 42, a 32-token cap, Q4_0 KV, flash attention, full GPU offload, two CPU threads, polling disabled and batch/ubatch 32. The same blocking-wait shim and 20% CPU quota were retained in both arms.

- CUDA baseline and its repeat produced byte-identical, valid single-object answers.
- Mojo produced the correct business/amount followed by repeated, incomplete JSON.
- Disabling CUDA graphs reproduced the exact same failing Mojo output. This rules out a failure specific to graph capture as the sole explanation.
- Shadow validation retained CUDA outputs and its valid answer. All 651 eligible calls passed elementwise checks. The runner initially expected 687 calls, so that run's raw status is retained as failed for missing the requested count; it did not fail numerically.
- All-call feedback validation reproduced the exact failing Mojo answer while **1119 calls / 10,885,632 values** passed the unchanged tolerance (maximum absolute difference 0.0000152587891). This supports accumulated numerical differences as a cause, but does not yet identify the precise arithmetic or first divergent logit.

[Quality results](../results/a2000/model-replacement/quality.json) and all completion text/counters are retained. A narrow quality check makes the regression reproducible:

```sh
python3 tools/check_completion.py results/a2000/model-replacement/completion-cuda.txt  # passes
python3 tools/check_completion.py results/a2000/model-replacement/completion-mojo.txt  # intentionally fails
```

The ordinary test suite passes by asserting that the bad output is rejected. The check accepts equivalent numeric/string amounts but rejects trailing text or a second JSON object; it does not mistake harmless formatting differences for this failure.

One feedback attempt was stopped by preflight at CPU 95°C before model startup; no model benchmark ran in that attempt. The guarded retry passed its numerical checks with peaks CPU 61°C/GPU 72°C. Other integration trials reached GPU 79°C, below the unchanged 82°C abort threshold. These brief trials do not establish improved thermal behavior. Sprite was resumed after every bounded pause.

Raw model-run summaries intentionally retain `model_sha256: null`: the runner did not rehash the model on these runs. The earlier fingerprint and exact filename/size are documented in [model-integration.md](model-integration.md). This is another reason not to promote smoke-test timings into a qualified model-speed claim.

## Reproduction and next gate

Build the observer and its C++ eligibility rejection tests with:

```sh
python3 tools/build_launch_observer.py
```

The runner accepts mutually exclusive `--validate-ptx` and `--replace-ptx` modes. `--feedback` requires validation, `--validation-calls 0` validates all eligible calls up to the hard experimental cap of 10000, and `--disable-graphs` supports the capture diagnostic. `--prompt` switches from `llama-bench` to greedy `llama-completion`; `--tokens` is an experimental run cap, not a PDF upload limit. `--wait-library` preserves the caller's local blocking-wait shim. Use the exact prompt recorded in the result summaries and the pinned `gate-order` artifact.

Next: capture matching-prefix logits and intermediate reductions to isolate the first numerical divergence; correct it without loosening quality checks, then rerun the quality and operation-speed gates. Only afterward run a matched whole-model throughput comparison and accurate PDF workflow. The present PR is evidence and investigation tooling, not deployment approval.
