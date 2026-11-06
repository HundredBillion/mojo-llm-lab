#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mojo_bin="${MOJO_BIN:-mojo}"
if [[ "$mojo_bin" == */* ]]; then mojo_bin="$(realpath -- "$mojo_bin")"; fi
case "${1:-}" in
  scalar) variant=q4-matvec; source=q4_k_q8_1_matvec.mojo ;;
  dp4a) variant=q4-dp4a; source=q4_k_q8_1_matvec_dp4a.mojo ;;
  dp4a-128) variant=q4-dp4a-128; source=q4_k_q8_1_matvec_dp4a_128.mojo ;;
  unrolled) variant=q4-unrolled; source=q4_k_q8_1_matvec_unrolled.mojo ;;
  packed) variant=q4-packed; source=q4_k_q8_1_matvec_packed.mojo ;;
  fixed) variant=q4-fixed; source=q4_k_q8_1_matvec_fixed.mojo ;;
  prepacked) variant=q4-prepacked; source=q4_k_q8_1_matvec_prepacked.mojo ;;
  *) echo 'Usage: build_q4.sh scalar|dp4a|dp4a-128|unrolled|packed|fixed|prepacked' >&2; exit 2 ;;
esac
mkdir -p "$repo_root/build/$variant"
cd "$repo_root/build/$variant"
"$mojo_bin" build --target-accelerator=nvidia:sm_86 --target-cpu=haswell \
  --emit asm -o host.s "$repo_root/kernels/$source"
