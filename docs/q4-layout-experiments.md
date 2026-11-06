# Fixed iteration and split-layout experiments

Neither new prototype beats the pinned CUDA operation.

| Paired experiment | GGML CUDA | Mojo | Qualification |
| --- | ---: | ---: | --- |
| Fixed two-step outer iteration | 95.455 µs | 108.464 µs | Original fixture, exact outputs |
| Split metadata/value layout | 116.256 µs | 131.024 µs | Varied-byte fixture, exact outputs, explicit upload synchronization |

The second candidate is about 12.7% slower by the ratio of medians. These are separate experiments with automatic clocks; do not infer a direct speedup between variants from their absolute times. Controls passed, and all 9728 output rows and memory canaries remained correct.

[Fixed-iteration data](../results/a2000/q4-cuda-fixed/) and [qualified split-layout data](../results/a2000/q4-prepacked-random/) include raw samples and hashes. [Initial split-layout data](../results/a2000/q4-cuda-prepacked-initial/) predates an explicit post-upload synchronization fix and the stronger fixture; it is retained as preliminary evidence, not the final qualification.

## What changed

`fixed` expands the remaining outer loop into two compile-time steps, guarded by the ten-block input-width bound. It retains the packed scale loads and 128-thread reduction.

`prepacked` additionally separates each Q4_K block's 16 metadata bytes from its 128 quantized-value bytes. Value blocks become 128-byte aligned. Metadata plus values occupy the same 144 bytes per block; the 9728-row matrix uses 14,008,320 bytes. Prepacking and uploads are outside the per-operation timer because model weights are static. The experimental harness retains both layouts for comparison; a production engine would need a load/prepack/upload strategy that fits VRAM without retaining two whole-model copies. That integration is not implemented.

The candidate still includes the same real GGML activation quantization kernel as the baseline. The baseline uses its normal interleaved weight layout. A new four-pointer ABI supplies metadata, values, activations and output. The harness synchronizes uploads before using its nonblocking compute stream.

## Stronger generated fixture

The original fixture varied scale/min fields but repeated the same quantized value pattern across blocks. A block-addressing mistake could therefore escape that case. The new generator preserves the metadata and activations but generates unique quantized bytes for all 640 source blocks, covering every byte value, then the benchmark repeats the 64 rows to its full shape. CUDA supplies the independent numerical oracle.

| Partition | Coverage |
| --- | --- |
| Normal dyadic blocks | Existing oracle and GPU comparisons |
| Representation-specific packing | All byte values, unique block values, both nibble halves, varied scale/min fields |
| Empty/malformed lengths | Existing CPU-oracle rejection tests; not a GPU-input fuzzing claim |
| General floating-point factors | Still outstanding; exact dyadic agreement does not establish this |

Seeds are 42 for the existing metadata/activation generator and 12345 for varied quantized values. The oracle property is preserved numerical output after changing the layout, not merely absence of a crash. The synchronized varied-byte run produced bit-identical outputs before and after timing.

## Register-pressure hypothesis

Driver-reported kernel attributes for the split-layout comparison were:

| Attribute | Mojo | GGML CUDA |
| --- | ---: | ---: |
| Registers per thread | 40 | 56 |
| Local-memory bytes | 0 | 0 |
| Occupancy API's maximum resident blocks per SM | 12 | 9 |

These are resource limits, not measured occupancy. They do not support register scarcity or local-memory spills as the immediate explanation for this candidate's slower timing. Read-only load behavior and block-reduction work remain worthwhile next experiments.

The qualified run peaked at CPU 59°C / GPU 66°C. Its short desktop pause ended after about 28 seconds, with an independent resume watchdog. Fixed and split kernel compilations peaked at CPU 74°C/GPU 66°C and CPU 79°C/GPU 67°C respectively. No production configuration changed.

## Reproduce the qualified split-layout case

Use the pinned dependencies, and keep external CPU/thermal protection active during compilation. The GPU runner has its own guards.

```sh
python3 tools/generate_q4_matvec_fixtures.py --random-quants --output build/q4-random
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh prepacked
python3 tools/build_q4_comparison.py --threads 128 --prepacked
python3 tools/run_q4_cuda_comparison.py --variant prepacked --fixtures build/q4-random --output results/local-prepacked-random
```

Use `fixed` with the ordinary 128-thread harness to reproduce that variant. The existing five-pointer fusion ABI remains the default validator contract; experimental three- and four-pointer kernels require explicit selection.

## Next falsifiable experiments

Keep the varied-byte correctness check, identical activation quantization, same-graph control and thermal guards for each candidate. Change one aspect per experiment:

1. **Reduction overhead:** replace the general block sum with a reduction that only produces the value for the output thread. Predict fewer barriers or shared-memory operations in emitted code, followed by lower paired operation time. Reject if numerical tolerance or canaries fail.
2. **Read-only loads:** explicitly select read-only loads for immutable weights and activations. First verify the emitted instructions differ; then measure whether that improves this access pattern. A source change with identical emitted loads is not a new performance experiment.
3. **Index width:** narrow proven-bounded offsets to 32 bits for this fixed shape. Predict fewer wide integer operations without altered addressing. This cannot be generalized to arbitrary matrix sizes without bounds checks.
4. **Thread assignment:** compare a 64-thread specialization with the 128-thread candidate using a matching launch configuration. Predict less idle work on the ten-block input width; measure the tradeoff against parallelism.
5. **Fused gate/up:** extend to the operation dominating the model trace, sharing activation loads across the two projections. This needs a new matching fused CUDA baseline and an independent correctness case; a standalone matvec ratio cannot establish its benefit.

These are hypotheses, not demonstrated improvements. Prefer an algorithm or fusion advantage over translating every instruction of an already tuned CUDA kernel. Any candidate must ultimately improve full-model generation or complete, accurate PDF processing to qualify as an application improvement.
