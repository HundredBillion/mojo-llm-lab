# Longer resident decode trials

Test whether the qualified 128-thread 2B Mojo kernel produces a measurable whole-model fixed-token decode improvement. Existing 16-token context-crossover results are mixed and CUDA/CUDA differences can exceed the expected few-percent effect. Add 128-token trials, retaining the existing 16/32-token defaults, model, thread settings, wait policy, context size, fixed per-context implementation and capture separation.

Extend only the synthetic fixture; assert that the first 42 token IDs match the original fixture so older short workloads remain comparable in content. A worker handshake must expose fixture length and maximum supported timed tokens. Reject insufficient or inconsistent capability metadata before requesting a trial. This remains teacher-forced decode, not prompt throughput, sampled-output quality or PDF success.

Use the unchanged AC/thermal limits and bounded normal-maximum-fan sessions. A predefined batch has controls before and after candidate assignments 1/0/0/1. Start with one ABBA cycle per run. Keep every failed attempt and stop on guards. No claim of success from trial duration alone; controls and candidate results must support an improvement. No permanent host settings change.
