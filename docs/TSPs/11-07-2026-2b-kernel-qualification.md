# 2B operation qualification plan

1. Parameterize fixture input width, preserving existing default bytes; test supported dimensions, quantization contract and distinct rows.
2. Parameterize native harness dimensions and build a separate 2B binary. Record actual dimensions.
3. Add explicit gate-order-2b runner variant with manifest, size and hash checks before GPU work and native result dimension checks.
4. Run CPU tests and compile under existing CPU/thermal controls. Run a guarded control/comparison with full distinct rows; retain artifacts and inspect correctness, control noise and timings.
5. Use results to choose launch geometry experiments; keep whole-model proof outstanding.

Routine design and plan review approved under workspace preference. No new thermal limits or model truncation.

## Geometry extension after baseline qualification

Two guarded baseline runs passed, with all 16 paired rounds favoring Mojo. Test 64 threads (two input blocks per projection lane) and 256 threads (128 threads each for gate and up), preserving dimensions and quantization. Keep variants in separate PTX and host binaries, validate expected entry and block size, and retain failures. Compare each against CUDA before any whole-model replacement. Different reduction order in the 64-thread variant must pass the same tolerance; do not relax it.
