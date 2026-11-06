# Research roadmap

The guiding target is a reproducible **2× improvement in both generation tokens/second and complete PDF parsing time at preserved quality**, with thermal behavior measured alongside speed. Any reproducible speed improvement counts as a success; 2× is a research direction, not a minimum acceptance threshold. Report both metrics separately, including regressions. The current result is an isolated Q-operation fusion; neither end-to-end gain has been demonstrated.

## 1. Define the whole-model baseline

Before another performance claim, record a benchmark contract covering:

- Both metrics: generation tokens/second and complete PDF latency, reported separately. Improvements to one must not be presented as improvements to the other.
- Exact engine revision/build, model and tokenizer revisions, weight quantization, projector if used, CUDA/driver versions and hardware.
- Prompt/input hashes, input and output token counts, context length, batch and slot settings, attention mode, KV-cache dtypes and GPU offload.
- GPU clock/power policy, CPU settings, fan mode, starting temperatures and external-power state.
- Quality expectations, numerical tolerances, completed-output requirements and the repeated-run protocol.

Use synthetic or explicitly redistributable inputs. Keep model downloads optional and pinned; the kernel test suite must remain usable without downloading a model. The benchmark contract and raw data should let another contributor reproduce the comparison without access to this project's development machine.

A candidate needs repeated, paired whole-model measurements against the same baseline. Report absolute times/rates, variability, failed runs and token counts. Changing models, reducing work or shortening outputs requires a separately labeled experiment.

## 2. Profile the actual model

Measure a complete decode and representative prefill before selecting the next kernel. Establish the share of time spent in quantized matrix operations, attention, memory movement and normalization/rotation. For PDF workloads, separate rendering, vision encoding, prompt processing and generation.

The current Q fusion saves roughly 1.9 microseconds per layer in the b12110 microbenchmark. That is useful evidence about this boundary, but an insufficient explanation for a 2× whole-model goal. Apply the measured time share before estimating an end-to-end benefit.

Prioritize work that removes substantial memory traffic, redundant transfers or dominant kernel time. Consider fusion, scheduling and existing engine implementations before introducing a new runtime or language boundary. A matched fused CUDA implementation would distinguish an algorithm/fusion gain from a language-specific advantage.

## 3. Integrate the first fusion carefully

Move from the fixed contiguous Q tensor to actual decoder buffers and streams. Verify strides, head counts, token batches, ownership and graph lifetime. Keep the runtime theta input: constant and runtime fast-math paths produced different long-position results during investigation.

Preserve numerical tests for position 32768, split-half pairing, the fourth-axis fallback, gamma-before-rotation and per-head normalization. Extend coverage before supporting K heads, other dtypes, KV-cache operations, different rotary modes or multi-token workloads. Keep unsupported configurations explicit and reject them.

An integration must reuse the existing CUDA context, stream and device buffers without unnecessary copies or a new host polling loop. Check blocking waits as well as GPU event time. Preserve all required model features, including vision features when the workload uses them.

## 4. Measure quality and thermal behavior

For each candidate, require:

- Numerical parity with the real engine reference under the predeclared criterion; never loosen a threshold simply to make a faster candidate pass.
- Complete workload outputs. PDF tests must preserve every page and transaction, including genuine duplicates, signed amounts and required schema fields. Original PDFs remain unchanged.
- Same-graph timing controls and repeated, interleaved candidate/baseline runs with recorded clocks and starting conditions.
- Temperature, power-source and abort records during sustained representative workloads. A short cold run does not establish laptop reliability.

Keep the runner's cold-start and stop thresholds. A guard stop is an incomplete result, not a performance win; do not raise thresholds to finish a run. Compare energy only when it is actually measured, and state whether it covers the GPU or the whole system. Do not infer joules or cooling improvements from a latency ratio.

## 5. Establish the end-to-end result

Only label a result “2× end-to-end” when the declared whole-model metric improves by at least 2× against its pinned baseline, on the full agreed workload, with retained quality and reproducible measurements. State the hardware, workload and limits alongside the claim.

Maintain kernel and whole-model results separately. Publish unsuccessful experiments when they help explain constraints. If the measured whole-model improvement is small, report that outcome and revisit the profile rather than promoting the microbenchmark ratio.
