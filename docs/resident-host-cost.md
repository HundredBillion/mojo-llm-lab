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
