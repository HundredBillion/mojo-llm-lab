# Packed integer instructions: first prototype timing

Explicit `dp4a.s32.s32` instructions improved the fixed-width Mojo prototype by **1.70× relative to its scalar Mojo version** in the public reproduction. This is not a comparison against GGML CUDA, a whole-model speedup, or a PDF parsing improvement.

| Arm | Median GPU time per 9728 × 2560 matvec |
| --- | ---: |
| Scalar Mojo | 669.120 µs |
| DP4A Mojo | 392.512 µs |

The same-graph control measured 649.774 versus 649.808 µs, a ratio of 0.999948. A preceding local run measured a 1.711× candidate ratio; the public runner reproduced 1.705×. [Raw public samples and hashes](../results/a2000/q4-dp4a-comparison.json) preserve the comparison.

## What changed

The original Mojo variant emitted scalar integer arithmetic. The candidate loads packed 32-bit words and executes four signed byte products per `dp4a` instruction. Q4 nibbles are masked into nonnegative byte lanes, while activation bytes remain signed. A second packed dot with four ones computes the integer sum used by the minimum-offset term. Emitted PTX contains two `dp4a.s32.s32` instruction sites inside loops.

Mojo exposes this through its documented [inline assembly interface](https://max.modular.com/stable/api/mojo/max/gpu/intrinsics/inlined_assembly/). No firmware modification or reverse engineering was needed. The optimized CUDA baseline already uses packed integer dot products, so this step brings the prototype closer to that baseline; it does not establish a language advantage.

## Method and limits

The 64 synthetic rows from the correctness fixture are repeated 152 times into a 14,008,320-byte weight buffer, yielding 9728 output rows. Each arm uses 32 serialized kernel nodes bracketed by GPU event nodes in a CUDA graph. Timing excludes host launch gaps, uploads, compilation and correctness checks. Eight ABBA rounds provide 16 measurements per arm. Warmup and the same-graph control precede the candidate comparison.

Both kernels produce exact outputs for this dyadic fixture, with intact output canaries before and after timing. GPU clocks were automatic; sampled clocks ranged from 1185 to 1252 MHz. This is not a fixed-clock experiment. Interleaving and the control help assess noise, but do not remove all thermal/clock confounding. The observed gain was much larger than the control discrepancy and reproduced in a second run.

The public run peaked at 75°C CPU / 77°C GPU. Compilation peaked at 83°C CPU / 67°C GPU. These short measurements do not establish sustained cooling or energy savings. The guard remained below its stop limits.

## Reproduction

After installing the pinned SDK and fetching upstream artifacts as described in the main README, generate inputs and compile both variants. Compilation also needs temperature monitoring and a bounded CPU quota; the build helper itself does not provide a thermal guard.

```sh
python3 tools/generate_q4_matvec_fixtures.py
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh scalar
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh dp4a
python3 tools/run_q4_comparison.py
```

The comparison requires a Linux systemd user manager, the NVIDIA driver, and GPU 0. It refuses active GPU compute, requires the existing cold/AC gate, limits its child to 20% CPU and 1 GiB host RAM, and stops its own scope on temperature, power or time failure. It writes under ignored `results/local-q4-comparison`; choose a new `--output` directory for another run. Do not call the worker directly: that bypasses the parent guard.

Remaining work: compare against a matched GGML CUDA timing baseline, optimize the grid and memory access, support the existing gate/up fusion, broaden numerical fixtures, then integrate and qualify generation and complete PDF workflows. No production setting has changed.
