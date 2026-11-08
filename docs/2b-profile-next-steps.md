# Choose the next kernel from the current model's profile

The 2B gate/up prototype has approximately 1.14× paired isolated-operation results, but the context crossover has not demonstrated whole-model improvement. The 64-thread experiment did not clearly improve on 128 threads. Increasing the number of tested layouts is not itself evidence of progress in end-to-end speed.

Even a repeatable operation gain can be small at model level. If the operation takes fraction f of the original runtime and improves by factor s, the idealized total speed ratio is 1 / (1 - f + f/s), assuming all other work and overhead stay unchanged. At s=1.14, hypothetical fractions of 10%, 20%, and 40% imply only 1.012×, 1.025×, and 1.052× total improvement. These fractions are illustrations, not measured attribution. The observed two-context control variability can obscure gains of that scale.

## Required diagnostic

Profile the pinned Qwen3.5-2B model with full GPU offload, two threads, polling disabled, explicit verified blocking CUDA waits, and the existing AC/temperature guards. Retain model/artifact identifiers, raw trace, launch dimensions, kernel counts and summed GPU durations. Separate warmup from timed work when the trace permits; disclose aggregation limits. Profiling overhead makes the trace unsuitable as a throughput claim.

The prepared local capture uses Nsight Systems with CUDA graph-node tracing, a 16-token decode and no CPU sampling. It requires both fans at least 4500 RPM for the temporary cooling experiment. A missing trace, rejected wait-policy receipt, fan-speed drop, cleanup failure or thermal stop must remain a failure, even if the profiler exits zero. The authenticated capture has now completed; see [the measured attribution](qwen35-2b-profile.md). This diagnostic still does not establish a throughput gain.

The older 4B attribution is not a substitute. The current model header contains Q6_K [2048,248320] token embeddings, Q6_K feed-forward down projections, and Q5_K attention projections; none is covered by the existing Q4_K gate/up replacement. Tensor size and quantization identify candidates, not their runtime cost.

## Candidate hypotheses, conditional on attribution

1. **Q6_K decode work dominates:** test row grouping and activation reuse, with the exact native quantization layout. The pinned CUDA implementation already uses packed integer dot products and explicit offset correction; a language translation alone has no established advantage. Preserve signed scales, rounding and matched Q8_1 inputs. Avoid expanding weights without measuring the added memory traffic and VRAM cost. [Pinned CUDA implementation](https://raw.githubusercontent.com/ggml-org/llama.cpp/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-cuda/vecdotq.cuh)
2. **Uncovered Q5_K or recurrent/attention work dominates:** target that measured operation before spending more effort on gate/up geometry. The successful gate/up format and dimensions cannot be reused indiscriminately.
3. **Host gaps or synchronization dominate:** investigate the measured gaps with matched CUDA controls; a kernel-only change may not affect the critical path. Keep host-efficiency improvements separate from claims of Mojo acceleration.

After a candidate passes numerical qualification, compare whole-model decode and prompt processing independently against the same tuned CUDA settings, then check sampled outputs and application validation. A 2× result remains an aspiration; smaller reproducible gains are useful, but neither microbenchmark wins nor passing CPU tests satisfy those whole-workload requirements.
