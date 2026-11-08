# Resident host-cost investigation

No new Mojo speedup is established. This follow-up instruments the boundary between untimed prefill and timed decode, then exposes the disposable container CPU quota as a bounded experiment parameter. The default remains 1000 µs per 5000 µs (20% of one core); 2000 µs is available but has not been tested or recommended.

## Instrumentation

The worker reports timed replacement and capture deltas alongside cumulative counters. The controller rejects missing, negative or inconsistent deltas. These are host submission/capture counts, not GPU graph replay counts. Timing, model, context, fixed tokens and synchronization remain unchanged. A new worker artifact means these results are not pooled with earlier resident runs.

## Completed control

A 16-token, one-cycle ABBA CUDA/CUDA control completed with Sprite running. Arm medians were 68.792 and 67.399 tokens/sec; ABBA latency ratio 0.97979. The approximately 2% difference is a same-implementation control observation, not an optimization gain. Both timed replacement/capture counts remained zero. The median reported throttled-time/wall-time ratio was 0.7962, which does not isolate critical-path latency.

Peak sampled temperatures were CPU 75°C/GPU 73°C/platform 64.05°C. Fixed-prefix top token 271 matched across trials. This short run neither establishes sustained thermal safety nor reproduces sampled PDF inference.

## Candidate blocked before launch

The subsequent Mojo preflight sampled CPU 93°C, GPU 64°C, platform 64.05°C with AC connected. The unchanged CPU 90°C guard rejected the run before any model container started. No candidate timing exists. A later process snapshot cannot establish the cause of that brief CPU spike. No higher-quota trial was attempted, and no host or the application defaults changed.

[Raw control and preflight failure](../results/a2000/resident-host-cost/) are retained. All owned inference containers were removed; unrelated containers and the original Sprite session remain running.

## Verification and remaining work

The worker compiled with warnings as errors under CPU and temperature guards. All 72 Python tests passed, including invalid timed-counter evidence. The first full-suite invocation used the wrong working directory and failed imports; rerunning from the repository root passed. The build peaked at CPU 61°C/GPU 58°C and the corrected test run at CPU 65°C/GPU 63°C.

Pending: candidate timed-capture observation; a matched repeat; only then a guarded quota comparison. The new counter instrumentation is verified on CUDA and unit-tested for candidate counter relationships, but the candidate live path remains unobserved. The original 4B isolated operation gain remains separate from these 2B whole-model experiments.

A focused review found an impossible timed-noncapture combination accepted by the validator. The invariant and a regression test now reject it; focused re-review found no remaining issue in that fix. The retained control also passes the updated validator.


## November 7 follow-up

Three attempts used the same worker, model, observer, PTX and wait shim at the default 20% CPU allowance. Sprite remained running. A separate half-second host sampler recorded temperatures and process CPU deltas; this observer itself adds a small CPU load, and these runs are not pooled with earlier data.

1. **Per-trial cooldown Mojo attempt:** both warmups completed; the next cold gate timed out before measured trials. The candidate warmup recorded 48 timed replacements, including 24 capture calls. This directly confirms capture activity inside that warmup's timed decode, not steady-state recapture behavior or graph replay counts. Peak CPU 83°C/GPU 72°C.
2. **Contiguous 16-token CUDA/CUDA control:** after initial and post-load cold gates, two warmups and one ABBA cycle ran without inter-trial cooldowns. It passed at CPU 77°C/GPU 79°C. Arm medians were 66.638/65.222 tokens/sec; latency ratio 0.97876. This is a distinct protocol from per-trial cooldown results and the earlier rejected 32-token block. Fixed-prefix top token 271 matched.
3. **Matching contiguous Mojo attempt:** the post-load cold gate sampled CPU 92°C/GPU 68°C and stopped before any trials. The worker had emitted its ready handshake; the model was loaded, then the owned container was removed. There is no candidate timing for this protocol. The higher CPU allowance remains untested.

The host recorder independently captured 92°C while a Chromium renderer used approximately 52% of one core over the preceding 0.52-second interval. A few seconds after model-container removal, it captured 94°C alongside another Chromium renderer at approximately 48% and 41% across two intervals. Device memory had returned to the 288 MB display baseline. These correlations identify browser work as another concrete investigation target; they do not prove a particular tab, exclusive cause, or a hardware diagnosis. The earlier November 6 spike is not retroactively attributed to Chromium.

[Raw attempts and selected host spike samples](../results/a2000/resident-host-cost/20261107/) are retained. Full host samples stay local. All owned model containers and the temporary sampler were stopped. No source/default/security setting changed during these follow-up experiments. The existing 72-test and C++ build evidence applies to the unchanged source; no fresh test run is claimed for this evidence-only update.

Next: identify the active renderer workloads and obtain a stable desktop comparison window before repeating the candidate. Separately, a controlled 40 Hz status-line fixture reproduced higher Sprite CPU than Alacritty, but fonts/cell grids were not normalized and stripped binaries prevented function-level attribution. Neither restarting Sprite nor changing model CPU limits has been shown to solve these issues.

The [verified wait-policy follow-up](resident-wait-policy.md) corrected an inactive helper target, completed controlled candidate runs and reduced CPU work. It supersedes the earlier assumption that preloading alone applied blocking waits; whole-model Mojo speed remains unproved.
