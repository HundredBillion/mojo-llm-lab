# Q6_K vocabulary projection: roofline before kernel work

Committed before the roofline measurement.

## Hypothesis

The largest single decode kernel in the [2B profile](../qwen35-2b-profile.md) is the Q6_K vocabulary projection: `mul_mat_vec_q<Q6_K>`, grid 248320, 24.08 % of summed decode GPU time. It reads a 248320 × 1680-byte weight matrix (417,177,600 bytes) per token. Its measured duration is 2.459 ms per launch in the decode trace and about 2.446 ms in the [prefill trace](../prefill-baseline.md) (313 ms / 128 calls), both at default clocks: an effective weight-read rate of about 170 GB/s. If achievable read bandwidth on this GPU is close to that, the CUDA kernel is already near the memory roofline, and **no standalone Mojo kernel can gain much**.

## Measurement

`kernels/stream_read_roofline.mojo` is a coalesced 8-byte streaming read of exactly 417,177,600 bytes with XOR accumulation, 128-thread blocks, at grids of 640, 1280 and 2560. Its output is verified exactly by per-thread parity. `tools/run_bandwidth_roofline.py` times it with CUDA events over 30 repetitions after 3 warmups, and also times a driver device-to-device copy as a read+write reference. The run uses default clocks, automatic fans and the standard cold-start, AC and abort guards. It takes well under a second of GPU work. The kernel is built with the existing guarded Mojo toolchain.

## Decision rule

Let `t_roof` be the fastest median stream-read time across grids, and `t_cuda` = 2.446 ms. The ceiling speedup for any vocabulary-projection kernel that must read the full weight matrix is `t_cuda / t_roof`.

- **Ceiling < 1.10:** a standalone Q6_K kernel can save at most `(1 − t_roof / t_cuda) × 24 %` of decode GPU time, under ~2.4 %. Do **not** build a standalone kernel. Record the result, and point further vocabulary-projection work at approaches that read fewer bytes (for example fusing or reducing the projection), not faster arithmetic.
- **Ceiling ≥ 1.10:** build a correctness-first Mojo Q6_K × Q8_1 kernel for this shape. Validate it against GGML's Q6_K dequantization semantics with full-output comparison, and compare it with the pinned CUDA kernel in a matched harness before any integration.

This probe bounds memory-bound kernels only. It is not a hardware-counter proof of saturation, and it says nothing about prefill MMQ work.
