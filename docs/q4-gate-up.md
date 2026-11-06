# Fused gate/up comparison

The first qualified fused Mojo prototype is near parity with the pinned CUDA operation. It is not a demonstrated speedup.

| Run | CUDA median | Mojo median | Baseline / candidate ratio |
| --- | ---: | ---: | ---: |
| First | 247.776 µs | 249.984 µs | 0.99117 |
| Repeat | 250.608 µs | 250.512 µs | 1.00038 |

The first run is about 0.9% slower; the repeat differs by only about 0.04% in Mojo's favor, too small to establish a gain given the observed variation. [First raw data](../results/a2000/q4-gate/) and [repeat](../results/a2000/q4-gate-repeat/) retain all samples and numerical logs. Same-graph control ratios were 0.99921 and 1.00136; both pass the unchanged 5% gate. Passing that gate does not establish statistical significance for a near-zero difference.

## Matched operation

The operation is `up * SiLU(gate)`, where each projection multiplies independent Q4_K weights `[2560, 9728]` by the same activation vector. Both arms include the identical real GGML Q8_1 quantization node and one fused projection/activation kernel. No separate CUDA projections are used as an artificially weak baseline.

The harness constructs `ggml_swiglu_split(gate_projection, up_projection)`. The pinned [fusion selection](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-cuda/ggml-cuda.cu#L4586) routes this graph to a fused matvec. Capture must have exactly two kernel nodes, and the matvec name must identify its one-column fusion-enabled specialization. Recorded CUDA name:

```text
_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj
```

The Mojo kernel uses two inlined projections, independent scalar block reductions, and SiLU. It retains native weight layout and ordinary loads from the qualified `reduce` variant. Its four pointers are up weights, gate weights, quantized activations and output. CUDA activation bytes are checked against the input fixture before comparison.

## Correctness and the rejected prototype

Gate weights have a separate byte seed and per-row factor signs/magnitudes. The independent CPU oracle checks exact zero with nonzero up output, positive/negative values near zero, and positive/negative saturation regions. Existing up-weight and activation fixtures are unchanged.

The initial prototype incorrectly passed a two-element SIMD pair to `block.sum`. That operation sums SIMD elements before reducing across threads, so it combined up and gate. The zero-gate row failed immediately: about 246.127 instead of zero. PTX confirmed the addition before reduction; the [library implementation](https://github.com/modular/modular/blob/main/max/mojo/max/gpu/primitives/block.mojo) also shows `val.reduce_add()`. This was a misuse of the reduction semantics, not evidence of incorrect CUDA arithmetic. The [rejected record](../results/a2000/q4-gate-rejected-vector/) is retained; no candidate timing was accepted.

Replacing that call with two scalar reductions passes the identical fixture. Warmup and timed checks use the original `absdiff <= 1e-5 + 1e-5 * abs(reference)` tolerance and intact canaries. Maximum absolute error is about **1.526e-5**, with 1824 of 9728 outputs differing bitwise; this is tolerance agreement, not bit-identical agreement. Nonlinear activation arithmetic means the earlier exact standalone-matvec result cannot be carried forward as an exact fused result.

The synthetic fixtures remain dyadic and repeat 64 source rows. General model activation distributions and real-model output quality are not yet qualified. Driver attributes report Mojo 38 registers/thread, CUDA 56, and zero local-memory bytes for both. Emitted PTX retains four block barriers in the Mojo candidate; a paired reduction that preserves independent sums is a concrete next optimization hypothesis.

## Conditions and scope

Both runs use 32 repetitions per graph, two warmup pairs and eight ABBA rounds, with automatic clocks/fans and the existing AC/thermal gates. GPU measurement peaks were CPU 62°C / GPU 72°C for the first control/comparison and CPU 69°C / GPU 70°C for the repeat. The corrected kernel compilation peaked at CPU 79°C / GPU 69°C. All temporary Sprite pauses had independent resume watchdogs; Sprite was resumed. No production setting changed.

This targets the fused operation prominent in the short model trace, but it is still an isolated benchmark. Token-generation throughput, complete accurate PDF latency and sustained energy/temperature improvements remain unmeasured for this candidate. No end-to-end speedup is claimed.

## Reproduce

Use pinned dependencies and external CPU/thermal guards during compilation:

```sh
python3 tools/generate_q4_matvec_fixtures.py --random-quants --gate-up --output build/q4-gate-fixtures
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh gate
python3 tools/build_q4_comparison.py --threads 128 --gate-up
python3 tools/run_q4_cuda_comparison.py --variant gate --fixtures build/q4-gate-fixtures --output results/local-gate
```

The emitted PTX also passed CUDA 12.8.1 `ptxas -arch=sm_86` validation before execution. `--gate-up` rejects prepacked layouts and launch widths other than 128, and the ordinary standalone harness remains the default.
