# Execution sequence

1. Record actual model shapes and pinned CUDA layout/fusion semantics.
2. Add an independent Q4_K decoder and edge-case CPU tests; compare it to the actual pinned GGML shared library.
3. Emit a Mojo GPU layout decoder and validate its ABI and outputs with guarded execution.
4. Add Q8_1 activation fixtures and integer-dot arithmetic, then standalone Q4_K matvec at actual shapes.
5. Add matched gate/up activation fusion; compare against the fused CUDA baseline with control runs.
6. Record gains or failures, integrate only justified candidates, and qualify both whole-workload metrics.

Stages 1–3 are correctness scaffolding and must not be described as an inference speed improvement. Existing compilation and GPU temperature limits remain in force.

## Reduction experiment

The fixed-iteration specialization emits three `bar.sync` instructions around its general block sum, although only thread zero consumes the result. Compare a separate `reduce` variant with only `broadcast=False` changed. Keep native weight layout, 128 threads, the same arithmetic, matched activation quantization, and varied-byte fixtures. Compiler acceptance on the pinned SDK and emitted PTX establish that the option is supported; numerical validation and guarded timing establish whether it helps. Keep the older variant as a reproducible comparator. Read-only loads and a hand-written reduction remain alternatives if this change is insufficient.
