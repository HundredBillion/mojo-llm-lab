# Implementation and verification

1. Add a small C++ worker using the exact pinned llama.h API, two contexts and serial stdin-controlled trials. Export observer counters for implementation-selection checks. Record fixed-prefix logits diagnostics separately from timed decode and CPU cgroup deltas.
2. Build with the pinned CUDA image and bounded CPU/memory. Use existing thermal monitoring. Add a Python host controller with initial, post-load and per-trial cold gates for the default short protocol, a total deadline, exclusive owned-container cleanup, raw logs, model/artifact identities and fail-closed result parsing.
3. Exercise CUDA/CUDA first, then CUDA/Mojo with identical settings and ABBA order. Keep all failed and accepted trials. Review whether variation permits a whole-model claim; do not equate teacher-forced decode with sampled responses or PDF latency.
4. Inspect the current the application PDF seam and run a disposable synthetic PDF case with known expected rows. Record text-only limitations and any integration failures without changing live documents.
