# Unrolling and packed scale loads

Two additional variants passed the matched CUDA harness. Neither beats llama.cpp yet.

| Paired experiment | GGML CUDA | Mojo | Mojo / CUDA time |
| --- | ---: | ---: | ---: |
| Compile-time inner-loop expansion | 95.647 µs | 117.181 µs | 1.225× |
| Expanded loops plus packed scale loads | 115.600 µs | 130.525 µs | 1.129× |

Both paths include the same activation quantization work and use the same 9728 × 2560 synthetic matrix as the earlier comparison. Outputs remained bit-identical over all 9728 rows, before and after timing, with intact canaries. Eight ABBA rounds produced 16 samples per arm; same-graph control ratios were 1.000137 and 1.000343.

[Unrolled raw results](../results/a2000/q4-cuda-unrolled/) and [packed-load raw results](../results/a2000/q4-cuda-packed/) preserve samples, artifact hashes and temperature peaks. These are separate runs, not a direct interleaved comparison between Mojo variants. Their absolute times vary, including the CUDA baseline; the smaller relative gap is encouraging, but is not sufficient evidence of a clean speedup of packed loads over unrolling alone.

## Changes and source evidence

The original kernel's two-iteration loops survived into generated PTX. The `unrolled` variant uses `comptime for` for the half and segment loops. This is supported by the installed Mojo 1.2.0 compiler (`2016d0cd`), consistent with the lockfile, and by the official [compile-time loop documentation](https://mojolang.org/1.1.0/docs/manual/metaprogramming/comptime-evaluation/) and [syntax migration notes](https://mojolang.org/releases/v0.26.2/).

The `packed` variant additionally reads three 16-bit packed scale/minimum words, extracts the two adjacent groups with masks, and reuses them across the expanded operations. The formula follows [the pinned GGML implementation](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-cuda/vecdotq.cuh); its MIT notice is retained under `licenses/`.

Static generated PTX contains 20 global-load instruction sites and 12 branch sites in the unrolled variant, versus 13 and 7 in the packed variant. These counts are intermediate-code observations, not measurements of hardware memory transactions or executed SASS instructions.

## Conditions and limits

Sampled graphics clocks were 1207 MHz, with automatic clocks and fans. Memory clocks were not recorded, so varying GPU memory states remain a possible confound to investigate rather than an established cause of timing variation. Add memory-clock observations and direct variant interleaving before attributing a stronger result.

The compile steps peaked at CPU 74°C/GPU 68°C and CPU 76°C/GPU 68°C respectively. Both timing comparisons peaked at CPU 67°C/GPU 68°C. A CPU-active desktop application was briefly paused under an independent 90-second resume watchdog; it resumed before the full jobs finished. No permanent system setting changed. These measurements do not establish sustained cooling or energy improvements.

The numerical corpus still uses dyadic factors. Broader floating-point fixtures, fused gate/up behavior, other shapes and full model/PDF accuracy remain necessary before integration. A smaller gap to CUDA is progress, not a whole-inference speedup.

## Reproduction

Use the existing pinned dependencies and synthetic fixtures. Apply external thermal/CPU protection to compilation; the GPU runner enforces its own guards.

```sh
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh unrolled
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh packed
python3 tools/build_q4_comparison.py --threads 128
python3 tools/run_q4_cuda_comparison.py --variant unrolled --output results/local-unrolled
python3 tools/run_q4_cuda_comparison.py --variant packed --output results/local-packed
```

The public runner never pauses user applications automatically.
