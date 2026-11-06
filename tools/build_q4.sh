#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mojo_bin="${MOJO_BIN:-mojo}"
if [[ "$mojo_bin" == */* ]]; then mojo_bin="$(realpath -- "$mojo_bin")"; fi
case "${1:-}" in
  scalar) variant=q4-matvec; source=q4_k_q8_1_matvec.mojo ;;
  dp4a) variant=q4-dp4a; source=q4_k_q8_1_matvec_dp4a.mojo ;;
  *) echo 'Usage: build_q4.sh scalar|dp4a' >&2; exit 2 ;;
esac
mkdir -p "$repo_root/build/$variant"
cd "$repo_root/build/$variant"
"$mojo_bin" build --target-accelerator=nvidia:sm_86 --target-cpu=haswell \
  --emit asm -o host.s "$repo_root/kernels/$source"
