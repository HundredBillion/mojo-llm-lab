# Mojo quantized matrix-vector investigation

## Problem and evidence

The short Qwen3VL trace attributed 87.84% of summed GPU kernel time to Q4_K/Q6_K matrix-vector kernels. The largest individual specialization is already fused: Q4_K, grid 9728, four warps per block, 612 launches, 133.976 ms. Source inspection shows that the specialization's fusion flag covers gate, bias or scale operations. Model shapes and launch counts strongly suggest the 36 feed-forward gate/up pairs, but the trace alone does not label tensor names.

The model has 72 Q4_K tensors with GGML dimensions `[2560, 9728]` (36 gate/up pairs), 36 Q4_K `[2560, 4096]` query tensors, and mixed Q4_K/Q6_K down/value tensors. Dimensions are input width then output rows. The output projection is Q6_K `[2560, 151936]`. Replacing only Q4_K leaves significant Q6_K work.

## Design choice

A full MAX engine migration would introduce model support, quantization and memory-fit uncertainty before isolating a gain. A floating-point dequantize-then-matmul replacement would add weight traffic and change the CUDA activation arithmetic. Start instead with a narrow packed Q4_K/Q8_1 GPU operation, followed by the existing gate/up fusion, against the pinned GGML implementation.

First validate the 144-byte Q4_K block layout (256 weights): two FP16 global factors, 12 packed scale/min bytes, 128 nibble bytes. CUDA quantizes activation vectors to Q8_1 and uses integer dot products; the candidate must match that path before performance claims. Keep a float64 mathematical decoder as an independent layout oracle, distinct from CUDA floating-point reduction parity.

## Requirements

- Synthetic inputs only; no model weights committed.
- Exercise all 6-bit scale/min values, both nibble halves, all eight subblocks, negative/zero factors, and multiple blocks.
- Verify decoder outputs against the pinned GGML decoder; then qualify GPU arithmetic before timing.
- Preserve gate/up fusion and output semantics in the dominant-shape comparison.
- Use realistic matrix dimensions, interleaved timing, canaries, warmup exclusion and thermal guards.
- Treat standalone layout/decode kernels as correctness scaffolding, not inference optimizations.
- Integrate only after repeatable gains; then measure generation and complete PDF accuracy/latency separately. Smaller credible gains are successes.

## Limits and decisions

This is inference kernel work, not GPU firmware replacement. No precision reduction, token truncation, omitted fields or removed vision features qualifies as a same-workload speedup. No change to permanent host settings is needed for the initial prototype.
