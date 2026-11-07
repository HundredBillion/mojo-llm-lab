# Validate full logits throughout the longer fixture

Check every vocabulary logit at prefill's final output, each priming output and each fixed decode step in the 2B fixture. Compare Mojo results with CUDA at the same position, including when Mojo is assigned to context zero and appears before the CUDA reference. Also compare repeated CUDA trials against their first reference.

Use a separate validation binary compiled from the resident worker. Extra synchronization, copies and comparisons invalidate timing; neither its elapsed times nor tokens/sec may be treated as performance evidence. Preserve the timing worker binary used by the preregistered comparison.

Before observing results, require finite values, matching argmax tokens, and absolute error <= 1e-4 + 1e-5 * abs(CUDA reference) for every compared logit. Require all expected positions and comparison counts, with no unpaired candidate snapshots. This tolerance is a bounded numerical criterion, not proof of arbitrary sampled-output equivalence or PDF accuracy.

Keep all thermal, power and cooling checks. Publish failures. A passing result strengthens fixed-fixture numerical evidence only; prefill performance and application quality remain outstanding.
