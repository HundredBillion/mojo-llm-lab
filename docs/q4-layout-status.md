# Q4_K prototype status

The experimental Mojo kernel in `kernels/q4_k_decode.mojo` decodes packed weights only. It is correctness scaffolding, not an optimized matvec or an inference integration.

The independent CPU oracle and the actual pinned GGML decoder agree exactly on 64 synthetic blocks (16,384 values), covering every 6-bit scale/min value and both nibble halves. The Mojo GPU decoder also agrees exactly on those fixtures, with intact output canaries. The factors in this suite are exactly representable dyadic values; this does not establish arbitrary floating-point rounding parity or dot-product reduction parity.

[Recorded results](../results/a2000/q4-layout-parity.json) contain source, kernel and PTX hashes. The bounded local build peaked at 85°C CPU / 66°C GPU; the subsequent GPU correctness check peaked at 68°C CPU / 64°C GPU. Neither measurement is evidence of an efficiency gain.

## Reproduce CPU layout checks

After fetching the pinned engine with the main README instructions:

```sh
python3 -m unittest discover -s tests -v
python3 tools/check_q4_cpu.py
```

## Compile the experimental GPU layout decoder

Keep this output separate from the established fusion benchmark's PTX directory:

```sh
mkdir -p build/q4-mojo
cd build/q4-mojo
../../.venv/bin/mojo build --target-accelerator=nvidia:sm_86 \
  --target-cpu=haswell --emit asm -o host.s ../../kernels/q4_k_decode.mojo
```

Use external temperature monitoring and a bounded CPU quota for compilation. Do not execute the host stub: its buffers are uninitialized. The emitted GPU entry accepts two device pointers (packed UInt8 blocks, FP32 outputs), one block of 256 threads per 144-byte input block. GPU parity was checked with a local CUDA driver launcher and a temperature-guarded process; a portable public GPU qualification runner is still outstanding. The existing fusion runner deliberately rejects this different ABI.

Next work is Q8_1 activation fixtures, packed integer dot products, actual model dimensions and gate/up fusion. The [design](PRDs/11-06-2026-quantized-matvec.md) and [sequence](plans/quantized-matvec.md) preserve those requirements.
