# Q6_K vocabulary projection is already at the memory roofline

Protocol: [`TSPs/11-07-2026-q6k-vocab-roofline.md`](TSPs/11-07-2026-q6k-vocab-roofline.md), committed in `24277b7` before measurement. Evidence: [`results/a2000/q6k-vocab-roofline/`](../results/a2000/q6k-vocab-roofline/).

The vocabulary projection, Q6_K `mul_mat_vec_q` over 248,320 rows, is the largest single decode kernel at 24 % of decode GPU time. It must read the whole 417,177,600-byte weight matrix for every generated token.

| Measurement (default clocks) | Time | Effective read rate |
| --- | ---: | ---: |
| Mojo stream-read of the same 417 MB, exact result verified, best grid (640 × 128) | **2.433 ms** median | 171.5 GB/s |
| Same, grid 1280 / 2560 | 2.438 / 2.449 ms | 171.1 / 170.3 GB/s |
| Driver device-to-device copy (read + write) | 5.059 ms | 164.9 GB/s combined |
| CUDA Q6_K vocabulary projection ([decode](qwen35-2b-profile.md) / [prefill](prefill-baseline.md) traces) | 2.459 / 2.446 ms | ~170 GB/s |

**Ceiling speedup for any kernel that reads the full matrix: 2.446 / 2.433 ≈ 1.005.** The pinned CUDA kernel already reaches about 99.5 % of the read bandwidth this GPU achieved for the identical byte volume. Under the preregistered rule (build only if the ceiling is ≥ 1.10), **no standalone Mojo Q6_K vocabulary kernel is built**. The best possible decode saving from one is about 0.1 % of decode GPU time.

Limits: the probe uses coalesced 8-byte loads. Wider 16-byte loads or other access patterns might reach somewhat higher bandwidth, but the CUDA kernel already matches this stream within 0.5 %, so the measurement does not support a meaningful kernel-level gain. Hardware counters were not used. The probe ran briefly at default clocks with automatic fans (peaks: CPU 82 °C, GPU 62 °C).

## Implication

Faster arithmetic cannot speed up the vocabulary projection; only reading fewer bytes can. Options that change bytes read, such as a smaller-vocabulary or differently quantized output head, or skipping the projection where the full distribution is unused, change model semantics or quality and need their own preregistered evaluation. The same reasoning applies to other matrix-vector kernels near this bandwidth: a Mojo kernel should target operations that are measurably below the roofline, or fuse work to remove memory traffic, as the gate/up fusion did.
