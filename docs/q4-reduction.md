# Avoiding an unused block-sum broadcast

The `reduce` prototype changes one operation from the native-layout `fixed` candidate: `block.sum[block_size=128, broadcast=False](partial)`. Only thread zero writes the output. The [documented API](https://max.modular.com/api/mojo/max/gpu/primitives/block/sum/) guarantees the complete sum in that thread when broadcast is disabled. The pinned Mojo 1.2.0 compiler accepts this specialization.

## Prediction and emitted code

The existing reduction distributes its result to all threads, although only thread zero consumes it. Disabling that distribution should remove synchronization and shared-memory work. The generated PTX confirms three block barriers become two, two shared stores become one, and two shared loads become one. The two remaining barriers are retained; this experiment does not hand-edit PTX or remove synchronization speculatively. [Instruction counts and PTX hashes](../results/a2000/q4-cuda-reduce/emitted-code.json) record the generated evidence, not runtime instruction counts.

## Results

Both comparisons include the same real GGML activation quantization, a native Q4_K weight layout, 128 threads, input width 2560 and 9728 output rows. The varied-byte fixture is used, with 32 repetitions per graph and two warmup pairs. Each run has its own same-graph control and eight ABBA rounds.

| Run | CUDA median | Mojo median | Mojo slowdown | Same-graph control ratio |
| --- | ---: | ---: | ---: | ---: |
| First | 106.318 µs | 113.408 µs | 6.7% | 1.00013 |
| Repeat | 120.335 µs | 129.600 µs | 7.7% | 1.00094 |

[First raw result](../results/a2000/q4-cuda-reduce/) and [repeat](../results/a2000/q4-cuda-reduce-repeat/) retain all samples. Both passed the existing 5% control gate, produced bit-identical outputs at warmup and after timing, and left memory canaries intact. Driver attributes report 39 registers per Mojo thread and zero local-memory bytes.

Absolute times vary appreciably, even though sampled graphics clocks were 1207 MHz. The two paired ratios consistently remain below one: Mojo has not beaten CUDA. Earlier broadcast-enabled results were collected separately; these data do not isolate the causal speed improvement versus that candidate. A directly interleaved old/new experiment would be needed for that claim.

The compile peaked at CPU 77°C / GPU 66°C. The two GPU experiments peaked at CPU 61°C / GPU 68°C and CPU 63°C / GPU 67°C. Independent-watchdog desktop pauses ended after approximately 20 and 19 seconds for timing; the build pause ended after 89 seconds. The first timing launch refused an existing output directory before GPU execution; it did not overwrite prior measurements. Sprite was resumed. No production configuration changed.

## Reproduction

Use the pinned SDK/dependencies and external compile-time CPU/thermal guards described in the repository. Generate the varied fixture before running:

```sh
python3 tools/generate_q4_matvec_fixtures.py --random-quants --output build/q4-random
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh reduce
python3 tools/build_q4_comparison.py --threads 128
python3 tools/run_q4_cuda_comparison.py --variant reduce --fixtures build/q4-random --output results/local-reduce
```

This remains a standalone synthetic matvec experiment. General floating-point input coverage, fused gate/up parity and end-to-end generation/PDF latency qualification are still outstanding. The next reduction question is whether a specialized implementation can safely avoid the remaining post-reduction barrier without changing numerical behavior; read-only loads remain an independent candidate.
