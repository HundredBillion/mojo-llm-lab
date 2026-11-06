# Execution sequence

1. Record actual model shapes and pinned CUDA layout/fusion semantics.
2. Add an independent Q4_K decoder and edge-case CPU tests; compare it to the actual pinned GGML shared library.
3. Emit a Mojo GPU layout decoder and validate its ABI and outputs with guarded execution.
4. Add Q8_1 activation fixtures and integer-dot arithmetic, then standalone Q4_K matvec at actual shapes.
5. Add matched gate/up activation fusion; compare against the fused CUDA baseline with control runs.
6. Record gains or failures, integrate only justified candidates, and qualify both whole-workload metrics.

Stages 1–3 are correctness scaffolding and must not be described as an inference speed improvement. Existing compilation and GPU temperature limits remain in force.
