# Read-only packed loads

This experiment did not establish an additional speed gain. The qualified explicit-load candidate measured **132.750 µs** against CUDA's **123.039 µs**, about **7.9% slower**. The reduced-broadcast candidate previously measured 6.7% and 7.7% slower in separate runs. These comparisons do not support preferring the read-only variant; they are not a directly interleaved comparison of the two Mojo kernels.

[Raw measurements](../results/a2000/q4-cuda-readonly/) retain the same-graph control, every timed sample, numerical-check log and assembler evidence. The control ratio was 0.999622, within the unchanged 5% gate. Warmup and timed outputs were bit-identical for all 9728 rows, and output canaries remained intact. This uses the varied-byte fixture, native layout, matched Q8_1 quantization and the same reduction as `reduce`.

## API code-generation finding

The first candidate used the documented [GPU load API](https://max.modular.com/stable/api/mojo/max/gpu/memory/memory/load/) with `read_only=True`. The pinned Mojo 1.2.0/MAX 26.7.0 AOT build completed but emitted malformed 32-bit loads, including `ld.global .nc` without a type suffix. NVIDIA CUDA 12.8.1 `ptxas` rejected sixteen instructions and exited 255. No GPU execution was attempted with that artifact.

The [unqualified source reproducer](../experiments/q4_readonly_api.mojo) and assembler errors are retained. This demonstrates an incompatibility in this exact source/toolchain path; it does not establish the root cause, or that every use of the API fails. The existing PTX ABI checker only validates the entry signature and target; it is not an instruction assembler. Compiler exit success and ABI validation alone do not establish loadable GPU code.

## Explicit-load fallback

The runnable `readonly` variant starts from `reduce` and changes only packed 32-bit weight and activation loads. Small inlined helpers emit `ld.global.nc.u32` and `ld.global.nc.s32`, using 64-bit pointer addresses and 32-bit result registers. Metadata and factor loads remain ordinary 16-bit loads. The assembly is conservatively marked side-effecting, which can constrain optimization. No precision, quantization or reduction-order change is intended.

The emitted PTX contains sixteen explicit read-only loads and the same two block barriers. CUDA's assembler accepts it. Driver attributes remain 39 registers/thread and zero local-memory bytes. Inputs are immutable during the candidate kernel; activation quantization is a preceding graph dependency. This assumption must be preserved by any future integration.

## Thermal scope and next direction

The successful fallback compilation peaked at CPU 83°C / GPU 69°C, below the existing abort limits. Its guarded GPU comparison peaked at CPU 61°C / GPU 69°C, and the timing pause ended after approximately 23 seconds. Sprite was resumed and no production setting changed. These short-run peaks do not establish sustained cooling or energy savings.

Next, prioritize the matched fused gate/up operation from the model trace. The pinned CUDA code already shares activation quantization and reduces both projections before applying the gate activation. A fair Mojo comparison must preserve that fusion and its nonlinear output semantics. A standalone matvec improvement would not by itself establish faster complete inference or accurate PDF processing.

## Reproduction

With the pinned SDK/dependencies and external compile guards:

```sh
python3 tools/generate_q4_matvec_fixtures.py --random-quants --output build/q4-random
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh readonly
python3 tools/build_q4_comparison.py --threads 128
python3 tools/run_q4_cuda_comparison.py --variant readonly --fixtures build/q4-random --output results/local-readonly
```

Before GPU execution, validate the generated PTX with CUDA 12.8.1 `ptxas -arch=sm_86 INPUT.ptx -o /dev/null`. The pinned CUDA development image in `versions.json` contains this assembler and needs no GPU access for that check.

To reproduce the disqualified API path, compile `experiments/q4_readonly_api.mojo` with the same `--target-accelerator=nvidia:sm_86 --target-cpu=haswell --emit asm` flags into a separate build directory, then assemble its emitted PTX. Do not use the normal GPU runner for that rejected artifact.
