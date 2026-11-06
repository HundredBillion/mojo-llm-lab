#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mojo_bin="${MOJO_BIN:-mojo}"
if [[ "$mojo_bin" == */* ]]; then
  mojo_bin="$(realpath -- "$mojo_bin")"
fi
build_dir="$repo_root/build/mojo"
mkdir -p "$build_dir"
cd "$build_dir"
"$mojo_bin" build --target-accelerator=nvidia:sm_86 --target-cpu=haswell \
  --emit asm -o host.s "$repo_root/kernels/fused_q_imrope_runtime_fastpow.mojo"
