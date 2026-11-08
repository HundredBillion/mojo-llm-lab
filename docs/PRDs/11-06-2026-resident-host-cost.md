# Separate host scheduling from Mojo decode gains

The existing resident comparison cannot distinguish a small candidate gain from CUDA/CUDA variation. The process consumes its 20% CPU allowance nearly every period; cumulative replacement counters also include untimed prefill. We need evidence that separates these effects before optimizing another kernel or changing defaults.

Chosen approach: add timed-section replacement/capture deltas and an explicit experimental CPU quota (1000 or 2000 microseconds per 5000 microseconds). Preserve the default 1000 quota, fixed workload, model, synchronization, cooldowns and all safety limits. First validate instrumentation at the current allowance; then compare CUDA controls at both allowances if a cool window is available. Only compare CUDA/Mojo at matching allowances. Retain failures and do not pool old/new worker binaries.

Alternatives: another Mojo kernel rewrite cannot resolve host confounding; removing CPU limits could raise heat without a controlled explanation. Both are deferred. CPU quota changes are confined to disposable containers and never become host or the application defaults from this experiment.

Acceptance: bounded quota choices and metadata; deltas strictly validated against cumulative counters; CUDA cannot claim replacements; timed capture distinguished from prefill; existing tests plus invalid-delta cases pass; compile with warnings as errors; preserve raw telemetry and guard stops. A speed claim requires repeatable matched controls and correctness, not merely a faster single sample.

Design review: counters measure host submissions/captures, not GPU replays. A capture delta of zero cannot prove every graph is reused. A quota comparison changes scheduling and heat together, so record both. No privileged operation, terminal pause, new server default, or live PDF job is required.


## November 7: primed decode comparison

Observation: candidate warmup submitted24 captured replacement calls inside the timed16-token decode. Add an explicit opt-in priming decode after the8-token prefill, synchronized before timing; timed tokens start at the next fixture token. CUDA and Mojo must use identical priming and final prefix. Keep original default protocol available. Record priming in worker handshake, trial rows and summary, and reject workload mismatches. A primed candidate that still captures inside timing is valid diagnostic evidence, but cannot be described as capture-free.

This is benchmark qualification, not an inference optimization. Alternatives—claiming old graph warmup removed capture, or raising quota before controlling desktop load—lack evidence. User authorized a temporary Chromium pause, in addition to prior Sprite permission. Use independent identity-checked resume watchdogs and a75-second experiment deadline within the90-second pause authorization; all thermal guards remain. Scope: one bounded matched block per process, then compare controls before any speed claim. Do not pool primed/unprimed records or silently discard failed attempts.

Two priming tokens are now selectable (default0, choices0/1/2). One left24 captures timed; two produced zero timed replacement/capture calls in the measured candidate. Validate the full prefill+prime+timed token count before each trial; the fixed Qwen fixture has49tokens. The timer excludes priming on both arms; this does not reduce actual request work.


## Verified CUDA wait policy

The existing preloaded helper ignores resident-decode unless LLAMA_CUDA_WAIT_TARGET names its absolute executable path. Add the explicit target and a selectable auto/blocking policy, with blocking as the resident harness default. Require exactly one matching constructor readback for device0 before marking a run passed. Include the existing helper source and a pinned, CPU-limited build recipe so reproduction does not depend on private machine files. Do not change host/the application defaults. Retain all untargeted measurements as historical, not verified-blocking evidence; compare matched policy controls separately from Mojo.
