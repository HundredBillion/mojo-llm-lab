# Qwen3.5-2B Mojo integration on A2000

The comparison uses the same Qwen3.5-2B Q4_K_M file for CUDA and Mojo, with pinned llama.cpp b12110. The model SHA-256 is `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`; see the [verified download](../results/a2000/qwen35-2b/download.json).

## Correctness gates

A separate `gate-order-2b` kernel handles 2048 inputs and 6144 output rows. Both input projections contain eight Q4_K blocks per row. The existing 2560/9728 specialization remains separate; the runner rejects swapping the named artifacts. Eligibility checks width, rows, strides, fusion features and the exact native specialization before replacement; unsupported calls retain CUDA.

Shadow validation and feedback validation each checked all 459 eligible calls, totaling 2,820,096 values per run. Maximum absolute error was 0.00000190734863, within the unchanged elementwise tolerance `1e-5 + 1e-5*abs(reference)`. Feedback copies checked Mojo outputs into subsequent model layers. Neither validation run measures performance.

The graph-enabled replacement then produced byte-identical CUDA answers for all three synthetic invoice examples: business/total, subtotal/tax/older paid invoice, and total versus remaining balance. These are limited text checks, not arbitrary-input equivalence or actual PDF/OCR validation. Raw [validation, output and sensor evidence](../results/a2000/qwen35-2b/mojo-specialization/) is retained.

## Reproduce

Build `tools/build_q4.sh gate-order-2b` using the pinned Mojo SDK, then rebuild the observer with `python3 tools/build_launch_observer.py`. Use the emitted four-pointer PTX entry from `build/q4-gate-order-2b/`.

Pass `--shape qwen35-2b` to `tools/run_launch_observer.py` for both arms. Set the same `--model`, `--wait-library`, prompt and token settings. Add `--validate-ptx <artifact> --validation-calls 0` for shadow validation, then add `--feedback` to test all checked outputs feeding forward. For graph-enabled execution use `--replace-ptx <artifact>` instead. Omit all PTX flags for CUDA.

For short decode measurements omit `--prompt`, use `--tokens 16 --repetitions 1`, and give every invocation a fresh output directory. The runner retains the same cold-start, AC and temperature guards. Never interpret validation timings as throughput, or zero replacements as successful Mojo execution.

## Same-model timing result: inconclusive

All 16 planned runs completed under the unchanged guards: four CUDA–Mojo–Mojo–CUDA cycles, one 16-token decode measurement per process. Pooled medians were **48.4161 tokens/sec CUDA** and **51.9263 tokens/sec Mojo**, a **1.0725×** ratio. This is an exploratory result, not a proved 7.25% improvement.

| Cycle | CUDA / Mojo mean decode latency | Direction |
| --- | ---: | --- |
| 1 | 1.3378× | Mojo faster |
| 2 | 0.8903× | Mojo slower |
| 3 | 1.0722× | Mojo faster |
| 4 | 1.0294× | Mojo faster |

The cycle directions disagree and individual measurements vary substantially. No samples were dropped. Model initialization, dynamic clocks and host scheduling under the shared CPU quota remain potential confounders; this experiment does not isolate their contributions. The [summary and every raw run](../results/a2000/qwen35-2b/mojo-specialization/decode-comparison.json) retain the evidence. GPU peaks were at most 73°C during these timing runs; this does not establish a sustained cooling or energy advantage.

The earlier 1.22–1.26× operation result applies to the **4B** specialization, not this new 2B kernel. Neither result proves PDF latency or a whole-model gain. A useful next timing design would reuse a loaded model, alternate separately captured CUDA/Mojo contexts, and measure CPU throttling during the decode interval, with CUDA/CUDA controls before interpreting small gains.

Verification: 66 Python regression tests passed; the C++ eligibility suite passed during the bridge build. Focused read-only review found no actionable issue in the new specialization; it did not independently exercise GPU workloads.
