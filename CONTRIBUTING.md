# Contributing

Contributions should make a performance claim easier to reproduce and assess. The current kernel experiment is narrow; broader inference improvements are welcome when they include a measured baseline and preserved quality.

## Start with CPU tests

```sh
python3 -m unittest discover -s tests -v
```

The tests use the standard library and synthetic data. Keep them independent of an installed SDK, Docker, CUDA and model downloads. Do not start GPU work from a test discovery hook or import.

Use the explicit GPU workflow in [README.md](README.md) when testing a device change. Do not remove thermal/power checks or run on a machine already showing unstable power or overheating. Describe any unsupported platform rather than assuming it matches the recorded A2000 configuration.

## Pull request evidence

A performance PR should answer these questions in its description:

1. What concrete operation or workload changes, and why should it take less work?
2. Which engine/model/compiler/driver versions and hardware were compared? Include relevant quantization, context, clock, power and fan settings.
3. What exact commands reproduce the baseline, candidate, correctness checks and timing control?
4. What are the absolute results, sample counts and variability? Attach compact raw samples and distinguish kernel GPU time from host time, generation throughput and complete workload latency.
5. Which outputs and numerical cases were checked? State the criterion before reporting the result and include failures or guard stops.
6. What remains unsupported, and what practical benefit has actually been demonstrated?

A correctness or tooling PR can use a shorter description with the relevant regression test and command. Keep unrelated changes separate.

## Numerical and timing rules

The existing GPU comparison uses `absdiff <= 1e-5 + 1e-5 * abs(GGML)`. Preserve it for the existing cases. Float64 mathematical tests do not replace an actual CUDA baseline comparison.

Run all three synthetic cases and the same-graph control for kernel results. Keep baseline and candidate graph structure comparable, GPU timing events inside the aggregate graph, transfers/JIT/warmup outside the timed interval, and serialized repetitions when they share output buffers. Do not infer a whole-model or energy gain from a kernel-only result.

Retain long-position and fallback-axis tests when changing fast math. The validated kernel receives theta through a runtime device buffer; substituting an apparently equivalent constant needs new evidence. Add layout/shape coverage before extending its supported interface.

## Result and dependency hygiene

Submit only synthetic or redistributable inputs. Do not include customer PDFs, account details, prompts containing private data, local absolute paths, credentials, device serial numbers or unrelated system logs. Compact generated inputs and hashes are preferable to large expected-value dumps.

Do not commit SDK environments, model weights, fetched engine/toolkit binaries, PTX, host assembly or local build output. Dependency changes must update pins and hashes with a reason and matching qualification evidence. Preserve upstream license notices and identify borrowed code or algorithms in the change description.

The repository's MIT license covers its own source. Separately fetched llama.cpp/GGML, Mojo/MAX and NVIDIA dependencies retain their upstream terms. Keep the project's source-only reproduction path intact.
