# Matched GGML CUDA comparison and work distribution

The explicit-DP4A prototype does **not yet beat llama.cpp**. Distributing quantized blocks across 128 threads narrowed the gap, but the current Mojo candidate remains about 1.62× slower in a matched standalone-operation comparison.

| Paired experiment | GGML CUDA | Mojo | Mojo / CUDA time |
| --- | ---: | ---: | ---: |
| 32-thread prototype | 94.556 µs | 387.728 µs | 4.10× |
| 128-thread prototype | 108.880 µs | 176.480 µs | 1.62× |

Each row is a separate interleaved comparison. The second experiment temporarily paused a CPU-active desktop application to obtain a cold window. Conditions differ between rows, as the CUDA times also show; do not treat their absolute Mojo ratio as a controlled scheduling-only speedup. The direct within-row result is the important finding: neither candidate beats CUDA.

[32-thread raw data](../results/a2000/q4-cuda-32/) and [128-thread raw data](../results/a2000/q4-cuda-128/) include controls and artifact hashes. Sampled graphics clocks were 1207 MHz in both paired experiments. Same-graph control ratios were 0.999664 and 1.004788 respectively. Clocks remained automatic; sampled graphics clocks do not establish fixed memory clocks or eliminate all power-state effects.

## Equal work and numerical checks

The harness uses the pinned b12110 GGML backend to construct and capture `mul_mat` for a 9728-row, 2560-input Q4_K matrix. The real CUDA graph contains two kernels: FP32-to-Q8_1 activation quantization and matrix-vector multiplication. The benchmark verifies the captured quantized bytes match the known synthetic fixture.

The Mojo arm includes a clone of that same quantization node and consumes its actual output pointer. Both timed paths therefore include activation quantization, use the same weights and quantized activations, and have two kernel nodes. This avoids crediting Mojo for precomputing input quantization outside the timer.

Each timed aggregate contains 32 serialized child graphs between GPU event nodes. Eight ABBA rounds provide 16 samples per arm; control runs time the same GGML graph in both arms. Uploads, capture, warmup and numerical checks are outside the timed interval. The original GGML graph and buffer ownership remain alive throughout replay.

Outputs were bit-identical across all 9728 rows before and after timing, with intact Mojo canaries. Inputs remain the synthetic dyadic fixture described in the earlier matvec report. Natural activation distributions, gate/up fusion, other input widths, prefill and complete model/PDF quality remain unqualified.

## Scheduling change

The first Mojo candidate used one 32-thread block per output row but only 16 active dot-product threads, each iterating all ten Q4_K blocks. The 128-thread version assigns each group of 16 threads a different quantized block, strides through the remaining blocks by eight, and reduces across the block. This follows the same broad work partition as the pinned CUDA kernel.

Packed instructions alone were insufficient: scheduling and memory access matter too. The next target is the remaining loop, load and scale-unpacking overhead, followed by matched gate/up fusion. Do not integrate a slower standalone prototype into the model based on its improvement over an earlier prototype.

## Reproduce

The pinned SDK and upstream archive from the main README are prerequisites. Keep CPU and thermal protection active during builds; the comparison runner itself enforces the cold-start and stop gates.

```sh
python3 tools/generate_q4_matvec_fixtures.py
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh dp4a
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh dp4a-128
python3 tools/build_q4_comparison.py --threads 32
python3 tools/build_q4_comparison.py --threads 128
python3 tools/run_q4_cuda_comparison.py --variant dp4a --output results/local-cuda-32
python3 tools/run_q4_cuda_comparison.py --variant dp4a-128 --output results/local-cuda-128
```

The GPU comparison requires standard NVIDIA CDI device 0 and Docker. It refuses existing compute work, validates the three-pointer PTX ABI, runs with bounded CPU/memory and no network, and cleans up only its own labeled containers. It does not pause any user application automatically.

The 128-thread experiment's maximum observed temperatures were 72°C CPU / 68°C GPU; its Mojo compilation peaked at 83°C CPU / 70°C GPU. Two harness-build attempts were deferred by the cold gate before the quiet window. The temporarily paused application was resumed after approximately 30 seconds, with an independent 90-second resume watchdog armed. No permanent fan, clock or Turbo settings changed.
