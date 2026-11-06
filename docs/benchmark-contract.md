# Whole-workload benchmark contract

The guiding target is 2× better generation throughput and 2× lower complete PDF latency. Smaller reproducible gains count as successes. Neither target has been demonstrated by the Q-operation microbenchmark.

## Two separate measurements

1. **Generation:** use the same model weights, tokenizer, prompt tokens, context occupancy, output length, and sampling policy. Report prompt processing separately from generation tokens/second. Include time to first token and total request time. Fixed-token synthetic decode is useful for diagnosis but does not prove answer quality.
2. **PDF:** measure from document submission to complete validated results, including extraction/rendering, vision when required, prefill, generation, validation and retries. Report native-text and scanned documents separately. Preserve every page, genuine duplicate transaction, signed amount, dates, balances and required metadata. A fast incomplete or incorrect result is a failure, not a speedup.

Both comparisons use pinned engine/model hashes and the same hardware and resource policy. A newer baseline should include already-established tuning so an avoidably slow default does not inflate gains. A configuration that cannot complete the accuracy corpus is not a qualified baseline for a success claim.

## Measurement sequence

- Profile a short synthetic run to choose targets. CUDA graph node tracing changes timing; never use profiled throughput as the headline baseline.
- Run unprofiled baseline and candidate in interleaved order after the same cold gate. Collect at least five paired observations and retain raw samples, errors and guard stops.
- Match GPU clock/power policy, CPU quota/affinity, thread counts, fan policy and starting temperatures. Record actual clocks and throttling. CPU-limited experiments establish results only under that limit; they do not establish unrestricted production throughput.
- Report paired ratios and absolute times, variability, token counts and correctness. A gain must exceed measurement noise. Report regressions in either metric even when the other improves.
- Keep temperature and AC guards active. Never raise stop thresholds to obtain a completed run. Short runs cannot establish sustained thermal reliability; energy claims require measured energy with a stated device/system boundary.

## Current gaps

There is no qualified whole-model Mojo integration yet. Earlier local PDF experiments exposed incomplete dates/balances under the full output schema; those are failed qualification runs, regardless of their latency. The initial public result remains an isolated, numerically validated Q fusion. Quantized matrix operations and attention need actual workload attribution before choosing the next Mojo kernel.

Public artifacts must use synthetic or explicitly redistributable documents. Keep private PDFs, full local profiler captures and machine-specific paths outside this repository; publish reviewed aggregate measurements and reproducible instructions instead.
