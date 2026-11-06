# First quantized matrix-vector correctness result

The experimental Mojo Q4_K/Q8_1 kernel now performs a standalone matrix-vector operation at input width 2560, matching the model's query and feed-forward input width. A 64-row synthetic fixture produced exact agreement among Mojo, a float64 mathematical oracle and the pinned GGML CUDA `mul_mat` operation. Output canaries around the Mojo allocation remained intact. [Recorded evidence](../results/a2000/q4-matvec-parity.json) includes fixture, source and PTX hashes.

This is **not a speed result**. The prototype uses scalar integer multiplies and a 32-thread block with 16 threads doing dot-product work. It does not yet use explicit packed dot-product instructions or implement the gate/up fusion. Those are the next optimization steps. A 64-row correctness fixture does not establish performance at the model's 9728-row feed-forward shape.

## Matching the activation arithmetic

The CUDA baseline internally quantizes FP32 activations into Q8_1. Each synthetic 32-element group has scale 1/128 and includes integer value 127, so its maximum magnitude divided by 127 reproduces the chosen scale exactly. Both implementations therefore consume equivalent quantized values. The stored Q8_1 sum is generated consistently, although this Q4_K dot-product path computes the relevant integer sums directly.

Q4_K factors are 1/512 and 1/1024; scale/min fields vary across all 6-bit values. The input generator is deterministic. These dyadic fixtures establish indexing, signedness, offset and accumulation correctness for the tested domain. They do not establish general floating-point reduction parity, natural model activation distributions, accuracy of generated answers or full PDF correctness. The declared check was `absdiff <= 1e-5 + 1e-5 * abs(reference)`; observed error was zero.

## Sources and checks

- `kernels/q4_k_q8_1_matvec.mojo`: fixed-width candidate; one grid block per output row, exactly 32 threads per block, three device pointers (weights, Q8_1 activations, output).
- `benchmark/q4_reference.c`: actual GGML CUDA reference, fixed 64-row shape. It initializes a blocking CUDA primary context before computation and rejects unmatched fixture scales.
- `tests/q4_reference.py`: independent mathematical decoder and dot-product oracle.
- `tools/generate_q4_matvec_fixtures.py`: redistributable synthetic fixtures; never uses model weights or private prompts.

```sh
python3 -m unittest discover -s tests -v
python3 tools/generate_q4_matvec_fixtures.py
mkdir -p build/q4-matvec
cd build/q4-matvec
../../.venv/bin/mojo build --target-accelerator=nvidia:sm_86 \
  --target-cpu=haswell --emit asm -o host.s ../../kernels/q4_k_q8_1_matvec.mojo
```

Use external temperature protection and a bounded CPU quota for compilation. The host stub is for AOT specialization only; do not execute its uninitialized inputs. The existing fusion timing runner cannot run this ABI. Local GPU qualification used a guarded CUDA driver launcher and containerized reference; publishing a portable runner for these experimental kernels remains outstanding.

Compilation peaked at 88°C CPU / 67°C GPU without crossing the 90°C stop threshold. The accepted Mojo/CUDA checks peaked at 78°C CPU / 66°C GPU. An earlier reference attempt stopped at its context-wait preflight; no numerical result was accepted from that attempt. No permanent clock, fan or Turbo settings changed.
