# Mojo LLM Lab

Experiments in making local LLM inference faster with Mojo GPU kernels, reproducible comparisons and preserved numerical accuracy. The guiding target is **2× faster token generation and complete PDF parsing**, with preserved quality. Any reproducible speed improvement is a success; 2× is not a minimum threshold. Neither end-to-end target has been achieved.

The first result is a fused RMSNorm, learned scale and IMROPE operation for one Q tensor. It runs about **1.92× faster than llama.cpp b12110 for this isolated operation** on an RTX A2000 Laptop GPU. There is no measured end-to-end LLM speedup or energy saving yet.

## Recorded result

Median GPU microseconds per operation, with 16 samples per arm:

| Synthetic positions | llama.cpp b12110 | Fused Mojo | Operation speedup |
| --- | ---: | ---: | ---: |
| Distinct axes | 3.979 | 2.072 | 1.92× |
| Position 32768 | 3.976 | 2.075 | 1.92× |
| Fourth-axis fallback | 3.984 | 2.079 | 1.92× |

[Raw measurements](results/a2000/) also include the older llama.cpp baseline, which measured approximately 4.216 versus 2.076 microseconds, or 2.03× for this operation. The newer baseline is the relevant comparison for current work.

Each sample launches one CUDA graph containing 128 serialized repetitions. Both implementations use the same child-graph structure, internal GPU timing events and ABBA ordering. A control that times the same baseline graph in both arms differed by approximately 0.2% with b12110 and 0% with the older baseline. Upload, warmup, host transfers and numerical checks are outside the timed interval.

The recorded runs used an RTX A2000 Laptop GPU with 4 GB VRAM at a sampled 1207 MHz GPU clock, an Intel i7-11850H and automatic fans. Observed CPU/GPU peaks were 73°C/66°C. These observations do not establish sustained cooling or energy improvements. All three fixtures produced bit-identical outputs in the recorded correctness checks, with intact output canaries. The benchmark also enforces `absdiff <= 1e-5 + 1e-5 * abs(GGML)`.

## What the kernel covers

The kernel processes one contiguous FP32 Q tensor: **32 heads × 128 dimensions**, one shared gamma vector of length 128, and four axis positions. IMROPE sections are `[24, 20, 20, 0]`; epsilon is `1e-6` and the frequency base is `5,000,000`. The fifth device pointer supplies the runtime FP32 theta-scale value `0x3f492c27`.

The baseline performs this chain in two kernels; this prototype uses one. The [b12110 fusion predicate](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-cuda/ggml-cuda.cu#L3133) supports NORMAL and NEOX modes but excludes IMROPE from this particular fusion. This is evidence for a missing fusion opportunity. It does not establish that Mojo is intrinsically faster than an equally fused CUDA implementation.

This is a hot-cache, single-operation microbenchmark. It does not cover complete model inference, PDF parsing, K heads, multi-token prefill, arbitrary GGML strides, other dtypes, YaRN or KV-cache writes. For illustration, saving about 1.9 microseconds in each of 36 Q branches would total about **69 microseconds per generated token** under matching conditions. Actual token impact has not been measured; the operation's 1.92× ratio cannot be applied to an entire model.

## CPU-only checks

The default tests need Python 3.12 or newer and its standard library and do not load CUDA, install an SDK or run GPU work:

```sh
python3 -m unittest discover -s tests -v
```

They check reproducible synthetic fixture hashes, binary layout, per-head normalization, gamma ordering, split-half rotation, long positions, fourth-axis fallback and rejection of unsupported PTX ABIs. The float64 mathematical oracle is distinct from the actual CUDA fast-math reference.

## Reproduce the GPU experiment

The GPU workflow is explicit. It requires Linux x86-64, Docker, a compatible NVIDIA driver, NVIDIA Container Toolkit with CDI configured, and a GPU supporting the `sm_86` kernel. The recorded platform used driver 615.78.08. Check [Modular's supported hardware and driver requirements](https://max.modular.com/gpu/hardware-compatibility/) before attempting a different configuration.

The versions and upstream archive/header hashes are pinned in [versions.json](versions.json). The CUDA development image is pinned by digest. The isolated SDK uses Mojo 1.2.0 and MAX 26.7.0; it is not installed globally.

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements.lock
python3 tools/fetch_upstream.py
python3 tools/generate_fixtures.py
MOJO_BIN=.venv/bin/mojo bash tools/build_mojo.sh
python3 tools/build_benchmark.py
python3 tools/run_benchmark.py --output results/local
```

The Mojo build emits host assembly and GPU PTX under `build/mojo/`; it does not execute the kernel's host stub. Generated fixtures contain only synthetic data. Upstream artifacts and generated files belong under `build/`, not in source control.

The runner selects GPU index 0 by default; use `--gpu` to select another supported GPU. It requires NVIDIA CDI access, a cold start below CPU 70°C/GPU 68°C, and AC power when a battery is present. It aborts at CPU 90°C or GPU 82°C. These guards reduce experimental risk; they do not diagnose hardware faults or guarantee a machine cannot overheat. Keep cooling settings unchanged during a comparison and record them with the result.

Do not treat a successful CPU test run as GPU qualification. Rebuilding on another compiler, GPU or driver requires the correctness and control runs again.

## Quantized matrix-vector investigation

The latest [fused gate/up candidate](docs/q4-reduction-order.md) measured **1.22× faster than the fused llama.cpp CUDA operation** in two guarded runs with distinct rows, varied FP16 weight factors and varied activation scales. It passed the unchanged numerical tolerance; outputs are not bit-identical. A regression comparison on the earlier full-row fixture measured 1.23×.

Broader inputs rejected the preceding `gate-pair` prototype despite its [earlier 1.08× result](docs/q4-full-rows.md). Matching CUDA's floating-point reduction order resolved that observed failure. Rejected cases and every timed sample remain published. The new `gate-order` kernel is an integration candidate; **no whole-model throughput, PDF-latency or energy improvement has been demonstrated**.

Earlier [standalone comparisons](docs/q4-cuda-comparison.md), [layout experiments](docs/q4-layout-experiments.md), [timing diagnostics](docs/q4-timing-diagnostics.md) and [read-only load investigation](docs/q4-readonly.md) document the unsuccessful approaches and remaining limits.

## Research direction and attribution

The first [whole-model kernel trace](docs/decode-profile.md) attributes 87.8% of summed GPU kernel time in a short synthetic decode to quantized matrix-vector operations. This identifies the next candidate; it is not a measured speedup. The [benchmark contract](docs/benchmark-contract.md) keeps generation throughput and complete PDF latency separate.

[ROADMAP.md](ROADMAP.md) defines the evidence needed before claiming a whole-model gain. [CONTRIBUTING.md](CONTRIBUTING.md) describes reproducible changes and result submissions.

This repository's code is licensed under [MIT](LICENSE). The benchmark uses separately fetched [llama.cpp/GGML source and libraries](https://github.com/ggml-org/llama.cpp), which retain their [MIT license](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/LICENSE). IMROPE semantics are grounded in its [CUDA implementation](https://github.com/ggml-org/llama.cpp/blob/d2e54583c7452353eb35d40431281f6ee984332f/ggml/src/ggml-cuda/rope.cu). Mojo/MAX and the NVIDIA CUDA toolkit remain subject to their respective upstream licenses; neither SDK nor CUDA binaries are vendored here. CUDA timing and graph behavior follow the [CUDA 12.8.1 graph API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__GRAPH.html) and [event API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__EVENT.html).
