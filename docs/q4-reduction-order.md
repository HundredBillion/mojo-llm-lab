# Reduction order: broader correctness and a larger operation gain

The `gate-order` candidate passes the broader fixture that rejected `gate-pair` and measures **1.22× faster than the fused CUDA baseline** in two guarded runs. A regression comparison on the previous full-row fixture also passes and measures 1.23×. These are isolated synthetic operation results, not token-generation or PDF-processing speedups.

| Candidate and input | CUDA median | Mojo median | CUDA / Mojo | Qualification |
| --- | ---: | ---: | ---: | --- |
| `gate-pair`, varied factors | 250.143 µs | 235.120 µs | 1.06390× | Passed |
| `gate-pair`, varied factors and activation scales | — | — | — | Rejected numerically before timing |
| `gate-order`, broader input | 267.680 µs | 219.072 µs | 1.22188× | Passed |
| `gate-order`, broader input repeat | 270.464 µs | 221.584 µs | 1.22059× | Passed |
| `gate-order`, original full-row input | 258.800 µs | 210.192 µs | 1.23126× | Passed |

Raw records: [factor-only](../results/a2000/q4-varied-factors/), [rejection](../results/a2000/q4-varied-activations-rejected/), [new order](../results/a2000/q4-gate-order/), [repeat](../results/a2000/q4-gate-order-repeat/), [original-input regression](../results/a2000/q4-gate-order-original/). All accepted runs pass the original 5% same-graph control gate. Controls for the two broader-input runs differed by approximately 0.13% and 0.78%. Every ABBA sample is retained.

## Expanded inputs

All matrices have 9728 distinct byte rows. Factor seed 739 independently multiplies each up-block FP16 factor by a value in [0.75, 1.25], then rounds to FP16; gate factors use seed 740. Packed scale/min fields and quantized values remain unchanged, and signs and zero gates are preserved.

Activation seed 971 independently scales each Q8 block by a power of two between 1/8 and 8, covering seven scales while preserving its signed quantized values. Cached sums are regenerated. The host now reconstructs FP32 inputs using the actual positive finite FP16 delta instead of assuming 1/128. Crucially, it still verifies that the real CUDA quantization node reproduces the expected Q8 bytes exactly before running the comparison. This is broader coverage, not arbitrary real-model activation coverage.

## Failure and fix

The first broader-input run failed at row 2069: Mojo produced approximately 6.07084703, CUDA 6.07075548. Their difference, about 9.155e-5, exceeded the unchanged allowed difference of about 7.071e-5. An independent float64 decoded-dot/SiLU calculation gives approximately 6.07080618, between the two. The [row diagnostic](../results/a2000/q4-varied-activations-rejected/row-diagnostic.json) records full values. No timing from this rejected run is accepted.

`gate-pair` reduced within each warp, then combined warp sums. The pinned [CUDA implementation](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-cuda/mmvq.cu#L785) first combines corresponding lane values across warps, then reduces the remaining warp. Floating-point addition order matters under cancellation.

The new `gate-order` variant follows that cross-warp-first order. Warps 1–3 write their lane partials for up and gate into separate shared-memory regions. All 128 threads meet one barrier; warp zero adds the three corresponding partials in order, then performs independent scalar warp reductions and SiLU. Scratch is not reused afterward. It retains one block barrier, while avoiding the preliminary reductions in the other warps. Generated PTX passes CUDA assembler validation.

The exact failing fixture now passes at warmup and after timing with the original tolerance and intact canaries. In the broader runs, maximum absolute difference is about 6.104e-5, with 1925 bitwise-different outputs out of 9728. This is tolerance agreement, not bit-identical agreement. The fix supports reduction order as the cause of the observed rejection; it does not prove all possible numerical inputs. Driver attributes remain 38 registers/thread and zero local-memory bytes for Mojo.

## Thermal record and scope

The first compilation attempt was stopped at a sampled CPU 95°C. A retry using a 10% CPU quota and CPU-0 affinity completed with a peak CPU 88°C / GPU 70°C. [Both build outcomes](../results/a2000/q4-gate-order/build-conditions.json) are retained. These are temporary build-process constraints; Turbo and system-wide settings were not changed. The independent Sprite watchdog resumed the desktop while the slower build continued.

The two broader-input GPU runs peaked at CPU 64°C / GPU 73°C and CPU 63°C / GPU 72°C; the original-input regression peaked at CPU 61°C / GPU 70°C. The factor-only run peaked at CPU 62°C / GPU 68°C. All AC/thermal limits remained unchanged, and Sprite was resumed. No production configuration changed. None of these short tests establishes sustained cooling or energy savings.

`gate-pair` should not be integrated on the strength of its earlier speed results: it failed this broader qualification. `gate-order` is the next integration candidate. Remaining work includes real-model tensor/output qualification, supported-shape fallback, token generation and complete accurate PDF latency. The primary user metrics remain both end-to-end measures; the 2× target is a guide, and smaller credible improvements count.

## Reproduce

Use pinned dependencies and external CPU/thermal guards during preparation/compilation:

```sh
python3 tools/generate_q4_matvec_fixtures.py --random-quants --gate-up --rows 9728 --factor-seed 739 --activation-scale-seed 971 --output build/q4-varied-activations
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh gate-order
python3 tools/build_q4_comparison.py --threads 128 --gate-up
python3 tools/run_q4_cuda_comparison.py --variant gate-order --fixtures build/q4-varied-activations --output results/local-gate-order
```

Use `gate-pair` on the same fixture to reproduce the rejected numerical case. Omit `--activation-scale-seed` for the factor-only case. Omitting both new seed flags preserves prior fixture behavior. CPU tests cover factor-byte isolation, signs/zeros, deterministic generation, activation-byte preservation, cached sums and invalid inputs; the GPU records establish the actual tolerance result.
