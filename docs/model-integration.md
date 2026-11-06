# Whole-model integration: launch observation

A guarded 16-token Qwen3VL-4B decode completed with a preload observer that **forwarded every kernel unchanged to CUDA**. It observed 72 host calls to the exact fused Q4_K specialization used by the microbenchmark. No Mojo model-kernel replacement or end-to-end speedup has been demonstrated yet.

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

1. Add eligibility checks for the exact shape, layout, activation and optional arguments, with CUDA fallback for everything else.
2. Validate Mojo against real-model projection outputs before replacing those outputs. Preserve CUDA graph semantics, context and stream ownership.
3. Verify model logits/generated outputs on synthetic requests, then benchmark an uninstrumented baseline and replacement under matched conditions.
4. Measure both generation throughput and complete accurate PDF processing. Keep production defaults unchanged until those checks justify adoption.

The existing isolated 1.22× operation result remains useful evidence for this next step, not a multiplier for the entire model.
