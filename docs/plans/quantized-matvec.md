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

## Read-only load experiment

Starting from `reduce`, replace only immutable input loads with `max.gpu.memory.load[read_only=True]`. The current PTX uses ordinary global loads; require the new build to emit different load instructions before benchmarking. Weights stay immutable, and Q8 activations are produced by a preceding graph node with an explicit dependency before the candidate reads them. Output remains a normal store. Preserve native layout, arithmetic, reduction, varied fixtures, correctness oracle and control gates. This tests a cache-access choice rather than a precision or workload change. Keep a separate variant so failures remain reproducible.

The pinned build accepted the high-level API source but emitted malformed 32-bit loads. NVIDIA `ptxas` rejected them before GPU execution. Preserve that source as an unqualified reproducer. The fallback tests only 32-bit packed weight/activation loads through explicit `ld.global.nc.u32`/`ld.global.nc.s32` helpers, retaining ordinary 16-bit metadata loads. Mark the assembly as side-effecting to prevent speculative removal or motion; this may limit optimization, so it is a distinct candidate rather than evidence about every read-only-load implementation.

## Fused gate/up qualification

Construct the pinned GGML graph as two Q4_K projections of one FP32 activation vector, followed by `ggml_swiglu_split(gate, up)`. Require capture of only quantization plus the fused matvec specialization; retain its name as evidence. The four-pointer Mojo candidate uses two inlined projections and independent scalar block reductions, then `up * SiLU(gate)`. Keep the proven native layout and ordinary loads rather than importing the unhelpful read-only variant. Use independent gate bytes and per-row factor signs/magnitudes so the nonlinear oracle sees zero, both signs near zero, and saturation regions. Preserve the existing tolerance, canaries, quantization-byte check and thermal gates. Initial goal is qualification; any performance claim is limited to this operation until integrated and measured end to end.

The first SIMD-pair reduction was rejected by the zero-gate GPU fixture before timing: it returned about 246.127 instead of zero. PTX showed an addition of up and gate before a single reduction. `block.sum` reduces SIMD lanes as well as threads, so it cannot express two independent sums that way. The corrected candidate uses two scalar reductions. Retain the failed numerical record, then rerun the identical fixture and unchanged tolerance to establish the fix.
