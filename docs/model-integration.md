# Whole-model integration: real-input validation

A guarded 16-token Qwen3VL-4B decode completed with a preload observer that **forwarded every kernel unchanged to CUDA**. It observed 72 host calls to the exact fused Q4_K specialization used by the microbenchmark. No Mojo model-kernel replacement or end-to-end speedup has been demonstrated yet.

## New: real-input differential validation

The opt-in validation bridge now checks every eligible launch's width, rows, strides, pointers, SiLU operation and absence of unsupported fusion features. It launches `gate-order` into separate storage on the current CUDA context/stream after the original kernel. CUDA outputs continue feeding the model. Both output sentinels must remain intact, every value must be finite, and every element must satisfy the unchanged `1e-5 + 1e-5*abs(reference)` threshold.

| Qualification | Compared calls | Compared values | Maximum absolute difference | CPU / GPU peak |
| --- | ---: | ---: | ---: | --- |
| Initial | 36 | 350,208 | 0.0000610351562 | 66°C / 63°C |
| Expanded | 612 | 5,953,536 | 0.0000610351562 | 61°C / 66°C |

Both passed. The expanded run validated all 612 observed eligible calls during the short 16-token `llama-bench` workload, including warmup. These use actual model weights and benchmark-driven activations; this is not a natural-language or PDF workload. This is bounded differential qualification, not exhaustive numerical proof, logit/generated-text equivalence, or an end-to-end speed measurement. Raw aggregate evidence and per-call error summaries are in [model-validation](../results/a2000/model-validation/full-612.json). Model tensors are not saved.

Graphs were explicitly disabled for validation, which allocates, synchronizes, copies outputs, and loads/unloads a module per checked call. Active capture is rejected. Those deliberately expensive operations must not be carried into a production replacement. The existing blocking-wait shim was preserved. Sprite resumed after approximately 13 and 21 seconds respectively. The model hash is intentionally null in these raw results: the runner did not rehash the model; the prior separate fingerprint below remains historical evidence.

```sh
python3 tools/build_launch_observer.py
python3 tools/run_launch_observer.py --model /path/to/model.gguf \
  --validate-ptx /path/to/gate-order.ptx --validation-calls 612 \
  --output results/local-model-validation
```

The PTX parser requires one self-contained sm_86 entry with four pointer parameters. The user must supply the qualified gate-order artifact; this is not a general arbitrary-kernel compatibility guarantee. The runner fails if requested calls, final counters, or finite numerical evidence are missing. Five new evidence-parser tests cover incomplete, duplicate, malformed and nonfinite records; the CPU suite now has 54 passing tests.

## Earlier: launch observation

[Recorded summary](../results/a2000/launch-observation/summary.json) contains the first four matching call descriptions, binary hashes, model fingerprint and thermal peaks. These are host launch calls, including warmup/capture. CUDA graph replays do not pass through these wrappers again, so the count is not a per-token kernel execution count.

## What the model actually passes

The first four matching calls all reported:

| Field | Observed value |
| --- | --- |
| Input width / output rows | 2560 / 9728 |
| Launch dimensions | Grid 9728×1×1, block 32×4×1 |
| Weight row stride | 10 Q4_K blocks |
| Activation column stride | 80 Q8_1 blocks |
| Gate pointer | Present |
| Gate operation | SiLU (`GGML_GLU_OP_SWIGLU`, enum value 2) |
| Routing IDs, biases, extra scales, shared-expert pointers | Absent |

This matches the specialization tested by `gate-order`. The observer only prints the first four matching calls; this is not evidence that every later call satisfies every optional-feature condition. Any replacement must validate its eligibility on every call and fall back to CUDA when conditions are unsupported.

The [pinned fusion ABI](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-cuda/common.cuh#L1592) is mirrored by the observer with a size assertion. It filters the exact mangled specialization and launch shape before reading scalar arguments. It prints no pointer addresses or tensor contents. NVIDIA documents [runtime function lookup and driver interoperability](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__DRIVER.html); future replacement must also respect the current CUDA context and stream.

## Loader issue found and fixed

The first run stopped before decode because `RTLD_NEXT` could not resolve `cudaLaunchKernel`. llama.cpp dynamically loads its CUDA backend/runtime locally. The observer now falls back to `dlopen` with `RTLD_NOLOAD` for the already-loaded `libcudart.so.12`, then resolves the real function through that handle. It does not create a second runtime instance. Both launch entry points are forwarded unchanged. The [initial failure](../results/a2000/launch-observation/initial-failure.json) remains recorded.

## Conditions and reproduction

The successful run used pinned b12110, Qwen3VL-4B-Instruct-Q4_K_M, full GPU offload, Q4_0 KV, flash attention, two CPU threads, polling disabled, batch/ubatch 32, zero prompt tokens and 16 generated tokens. It used a 20% CPU quota and the existing local blocking-wait library alongside the observer. Peaks were CPU 59°C / GPU 70°C. Sprite was resumed after the bounded pause; no production configuration changed.

The model was fingerprinted afterward in a separate guarded step:

```text
bytes: 2497281664
sha256: 66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
```

That hash step used a 10% CPU quota and peaked at CPU 81°C / GPU 67°C. The runner does not yet compute the model hash itself; its raw result explicitly leaves that field null. The published summary annotates the separately acquired fingerprint. Future performance comparisons must pin and verify the model before accepting results.

```sh
python3 tools/build_launch_observer.py
python3 tools/run_launch_observer.py --model /path/to/model.gguf --output results/local-observation
```

The runner requires Docker, standard NVIDIA CDI device 0 and the pinned engine artifacts. `--wait-library /path/to/library.so` optionally mounts the caller's existing blocking-wait shim and records its hash; it is not distributed by this repository. Reproducing the recorded host-wait behavior requires the corresponding shim. The observer's metadata/layout result is not a performance comparison with or without it.

## Next implementation gates

1. Implement a separate opt-in replacement mode with cached module ownership and CUDA graph capture/replay support; retain CUDA fallback for unsupported calls.
2. Extend real-input qualification to natural-language prompts and then validate model logits/generated outputs with candidate outputs feeding forward.
3. Verify model logits/generated outputs on synthetic requests, then benchmark an uninstrumented baseline and replacement under matched conditions.
4. Measure both generation throughput and complete accurate PDF processing. Keep production defaults unchanged until those checks justify adoption.

The existing isolated 1.22× operation result remains useful evidence for this next step, not a multiplier for the entire model.
