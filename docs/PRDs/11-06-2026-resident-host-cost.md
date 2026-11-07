# Separate host scheduling from Mojo decode gains

The existing resident comparison cannot distinguish a small candidate gain from CUDA/CUDA variation. The process consumes its 20% CPU allowance nearly every period; cumulative replacement counters also include untimed prefill. We need evidence that separates these effects before optimizing another kernel or changing defaults.

Chosen approach: add timed-section replacement/capture deltas and an explicit experimental CPU quota (1000 or 2000 microseconds per 5000 microseconds). Preserve the default 1000 quota, fixed workload, model, synchronization, cooldowns and all safety limits. First validate instrumentation at the current allowance; then compare CUDA controls at both allowances if a cool window is available. Only compare CUDA/Mojo at matching allowances. Retain failures and do not pool old/new worker binaries.

Alternatives: another Mojo kernel rewrite cannot resolve host confounding; removing CPU limits could raise heat without a controlled explanation. Both are deferred. CPU quota changes are confined to disposable containers and never become host or the application defaults from this experiment.

Acceptance: bounded quota choices and metadata; deltas strictly validated against cumulative counters; CUDA cannot claim replacements; timed capture distinguished from prefill; existing tests plus invalid-delta cases pass; compile with warnings as errors; preserve raw telemetry and guard stops. A speed claim requires repeatable matched controls and correctness, not merely a faster single sample.

Design review: counters measure host submissions/captures, not GPU replays. A capture delta of zero cannot prove every graph is reused. A quota comparison changes scheduling and heat together, so record both. No privileged operation, terminal pause, new server default, or live PDF job is required.
