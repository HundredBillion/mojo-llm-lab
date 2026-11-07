# Resident logit validation plan

- Implement `integration/logit_validation.h` as a CPU-testable position-indexed reference comparator. Support either context order, reject missing/count-mismatched data, and summarize candidate and repeated-CUDA comparisons separately.
- Compile validation instrumentation into a separate `resident-logits` binary from `integration/resident_decode.cpp`; leave the performance binary unchanged. Observe after synchronization at every fixture position.
- Add `--resident-logits` to the guarded build and `--validate-logits` to the controller. Require Mojo mode, mark the run numerical-only, parse the mandatory report, and reject attempts to summarize it as performance.
- Test tolerance failures, missing positions, candidate-first pairing and nonfinite input. Build and run under existing CPU/thermal limits; validate both context assignments if the cooling window permits.
