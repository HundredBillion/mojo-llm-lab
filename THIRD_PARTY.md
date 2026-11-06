# Third-party components

This repository's source is MIT licensed. It does not redistribute the Mojo SDK,
NVIDIA CUDA toolkit, llama.cpp binaries, model weights, or private documents.

- [llama.cpp / GGML](https://github.com/ggml-org/llama.cpp) is MIT licensed.
  The benchmark downloads pinned headers and an official binary release; their
  upstream license remains applicable. The scalar reference follows GGML's
  RMSNorm and interleaved multi-axis RoPE semantics.
- [Modular Mojo / MAX](https://www.modular.com/legal) provide the compiler and
  standard libraries. Their own licensing and distribution terms apply.
- [NVIDIA CUDA](https://docs.nvidia.com/cuda/eula/index.html) and the NVIDIA
  container image retain their NVIDIA licensing terms.

Downloaded artifacts and generated binaries remain in ignored `build/` or
`.venv/` directories. Dependency versions and checksums are in `versions.json`
and `requirements.lock`.
