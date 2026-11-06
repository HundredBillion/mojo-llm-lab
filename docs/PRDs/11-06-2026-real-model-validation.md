# Real-model qualification of the fused gate/up candidate

The isolated gate-order candidate passes synthetic correctness checks and improves operation latency by about 1.22×. Integration must establish whether the same numerical contract holds on actual model weights and activations before consuming candidate outputs.

Use an opt-in launch interceptor with the pinned b12110 fusion ABI and exact symbol, shape, stride, activation, and optional-feature eligibility checks. Unsupported calls retain CUDA. The validation path first executes CUDA, then executes Mojo into separate guarded storage on the same current context and stream. Compare every result with the existing `1e-5 + 1e-5*abs(reference)` tolerance, reject nonfinite values, and check both buffer sentinels. CUDA outputs remain in the model.

Alternative approaches are an immediate backend replacement or offline tensor capture/replay. The interceptor is selected because it exercises real inputs without feeding unqualified outputs forward or saving model tensors. Immediate replacement would mix numerical and integration errors; replay would not establish live stream/context interoperability.

Validation explicitly disables CUDA graphs and rejects active capture. This permits allocation, synchronization and host comparisons. It is not a performance test. A later replacement must avoid these operations in its hot path and preserve graph capture/replay.

Implementation sequence:
1. Extend the observer with per-call eligibility and opt-in differential validation.
2. Run a 36-call initial qualification under existing thermal and AC guards.
3. Expand to all matching calls in the short decode; retain failures without loosening tolerance.
4. Test evidence parsing against partial, duplicated, malformed and nonfinite records.
5. Publish aggregate evidence without model tensors, binaries, or private paths.

Acceptance here does not establish generated-text/logit equivalence, end-to-end speed, PDF accuracy, or reduced energy. Those are subsequent gates for production adoption. Routine design/implementation approval is covered by the user's standing workflow preference.
