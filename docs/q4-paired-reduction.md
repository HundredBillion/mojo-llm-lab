# Paired scalar reduction: measured fused-operation gain

The `gate-pair` prototype measured **1.05–1.06× faster** than the pinned fused CUDA baseline in two guarded synthetic comparisons. This is the first repeated gain in this repository's fused Q4_K gate/up investigation. It is not a whole-model or PDF-processing speedup.

| Run | Fused CUDA median | Fused Mojo median | CUDA / Mojo |
| --- | ---: | ---: | ---: |
| First | 250.751 µs | 236.528 µs | 1.06013× |
| Repeat | 252.960 µs | 241.008 µs | 1.04959× |

[First raw record](../results/a2000/q4-gate-pair/) and [repeat](../results/a2000/q4-gate-pair-repeat/) retain every sample and correctness log. Same-graph controls differed by about 0.27% and 0.45%, within the unchanged 5% gate. The paired ABBA diagnostic ratios were 1.07958 and 1.04632. One round in the repeat favored CUDA; it is retained, not removed. Two runs support a gain on this fixture and platform, not a universal or sustained-workload claim.

## Change and synchronization argument

The [previous fused candidate](q4-gate-up.md) used independent block reductions and was near parity. This variant preserves the projections, arithmetic and nonlinear activation, but explicitly coordinates the two sums:

1. Each of four warps independently reduces its up and gate partials.
2. Each warp's lane zero writes its two results into separate four-element shared-memory regions.
3. All 128 threads meet one barrier.
4. Warp zero reads the eight initialized slots, pads the remaining lanes with zero, and independently reduces both sums. Thread zero writes `up * SiLU(gate)`.

The scratch is used once in the kernel and is never overwritten after that barrier. Other warps can finish while warp zero reads it; shared storage remains block-scoped. There is no later shared-memory reuse requiring a second barrier. This is deliberately separate scalar reduction, avoiding the SIMD-lane summation bug found in the earlier prototype.

The pinned compiler emits **one block barrier instead of four**, and CUDA 12.8.1 `ptxas` accepts the result. [PTX hashes and static counts](../results/a2000/q4-gate-pair/emitted-code.json) preserve that evidence. Driver attributes remain 38 registers/thread and zero local-memory bytes for Mojo, versus 56 registers/thread and zero local-memory bytes for CUDA. These are resource attributes, not measured occupancy. Earlier candidate times were recorded separately; a direct old/new comparison would be needed to isolate the size of the reduction change's benefit independently of other run conditions.

## Correctness, baseline and limits

The CUDA baseline remains its fusion-enabled Q4_K gate/up specialization with one shared activation-quantization kernel, verified by graph node count and recorded specialization name. Both arms consume the same quantized activation bytes. The dimensions are input width 2560 and 9728 output rows.

Both runs pass the original `absdiff <= 1e-5 + 1e-5 * abs(reference)` tolerance at warmup and after timing; output canaries stay intact. Maximum absolute error remains approximately 1.526e-5, with 1824 bitwise-different outputs out of 9728. This is tolerance agreement, not bit-identical agreement. The fixture covers zero gate with nonzero up, both signs near zero, and saturation regions.

The fixture uses power-of-two quantization factors and repeats 64 source rows. Before integration, expand to full distinct rows and varied factors/activations; then qualify model outputs and both token generation and complete accurate PDF latency. No data are available yet for this candidate's end-to-end impact, sustained energy use or cooling benefit. No production configuration changed.

## Conditions

Both runs use 32 repetitions per timed graph, two warmup pairs, eight ABBA rounds, and automatic clocks/fans. Point-sampled graphics clocks were 1207 MHz; full clock samples remain in the raw records. Compile peaks were CPU 78°C / GPU 68°C. GPU-test peaks were CPU 63°C / GPU 70°C for the first run and CPU 61°C / GPU 70°C for the repeat. Thermal/AC guards remained unchanged. Sprite was resumed after the short timing pauses, with independent watchdogs also armed.

## Reproduce

Use pinned dependencies and external CPU/thermal protection for compilation:

```sh
python3 tools/generate_q4_matvec_fixtures.py --random-quants --gate-up --output build/q4-gate-fixtures
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh gate-pair
python3 tools/build_q4_comparison.py --threads 128 --gate-up
python3 tools/run_q4_cuda_comparison.py --variant gate-pair --fixtures build/q4-gate-fixtures --output results/local-gate-pair
```

The variant retains the same four-pointer ABI and uses the same fused host harness as `gate`. The reduction uses the SDK's [shared stack allocation](https://mojolang.org/docs/std/memory/stack_allocation/unsafe_stack_allocation/) and scalar warp operations, with correctness checked on the pinned SDK rather than inferred from a source-level change alone.
