# Verified wait policy and primed decode

**No whole-model Mojo speedup is established.** This investigation fixed an inactive CPU-efficiency setting in the benchmark harness and separated replacement capture from timed decode. The original 4B isolated operation result remains a different claim.

## Root cause and fix

The preloaded wait helper only selects `llama-server` by basename unless `LLAMA_CUDA_WAIT_TARGET` supplies an exact absolute executable path. The resident worker had a different name and supplied no target. Preloading the helper and recording its hash therefore did not establish that blocking waits were active. Historical resident logs have no verified-policy receipt; they remain published, but must not be described as verified blocking-wait measurements.

The resident controller now supplies `/observer/resident-decode`, requests an explicit `auto` or `blocking` policy, and requires a matching device 0 startup readback before marking a run passed. Blocking is the harness default. The shared receipt validator rejects missing, duplicate, wrong-device and wrong-policy evidence while preserving unrelated flag bits. This proves constructor-time driver readback; it is not continuous monitoring of runtime policy.

A sibling inspection found the same omission in `run_launch_observer.py`. When a wait library is explicitly provided, that tool now targets `/app/llama-completion` or `/app/llama-bench` and requires a blocking receipt. The server PDF harness already uses the recognized basename. Neither host-wide nor the application defaults changed.

The helper source is included in `integration/cuda_wait.c` with a pinned, CPU-limited build in `tools/build_wait_policy.py`. It does nothing to untargeted processes. The newly built artifact passed a separate resident CUDA control with a verified blocking receipt; it has a different library hash and is not pooled into the installed-helper timing comparison. The two legacy CLI entrypoints were inspected, not newly exercised live in this follow-up.

## Priming experiment

One untimed single-token decode still left 24 captured replacements inside each measured candidate trial. Two synchronized priming decodes reduced timed replacement and capture counts to zero in the retained candidate trials. This distinguishes preparation from subsequent full-model fixed-token decode; it does not remove actual request work or count GPU graph replays.

Priming is explicit (`--prime-tokens 0|1|2`, default 0), recorded in the handshake/trials/summary, and validated. Each trial checks that prefill+priming+timed tokens fit the tokenized fixture. Earlier one-token exploratory records use the transitional `prime_decode` field; the summarizer labels them as one priming token. Different worker artifacts, priming counts and verified policies are not pooled.

## Matched wait-policy observations

All five runs below share the same worker/model/observer/PTX/wait-library artifacts, 20% CPU quota, two priming tokens, two warmups and one 16-token ABBA cycle. Medians cover four measured trials per run. The control's two arms both execute CUDA. The candidate places Mojo in arm 1. Model decoding uses fixed teacher-forced tokens; it is not sampled chat, prompt throughput or PDF latency.

| Run | Median CPU time/trial | Median decode tokens/sec | Median throttle counter / wall time | CPU/GPU peak |
| --- | ---: | ---: | ---: | --- |
| Automatic wait control 1 |43.903ms |72.63 |79.55% |71/78°C |
| Blocking wait control 1 |14.557ms |76.47 |4.04% |67/78°C |
| Blocking wait CUDA/Mojo |15.939ms |75.89 |3.96% |62/78°C |
| Blocking wait control 2 |14.216ms |79.82 |1.71% |65/77°C |
| Automatic wait control 2 |43.416ms |73.52 |79.81% |73/79°C |

These short exploratory controls show approximately 67% less CPU work per timed trial with blocking waits. The reported throttling counter is not a causal decomposition of runtime. Median throughput did not fall in these controls, but the sample is too small and noisy for a general throughput or sustained cooling claim.

Ratios divide arm 0 latency by arm 1 latency. The blocking CUDA/Mojo ABBA latency ratio was 0.91593: its Mojo arm was slower in that run. Blocking CUDA/CUDA controls also favored arm 0, with ratios 0.90528 and 0.96466. This prevents attributing the observed difference solely to Mojo, in either direction. Sampled GPU clocks varied; these are not fixed-clock experiments. Fixed-prefix top token 4952 matched within all two-prime runs, a narrow check rather than full-output equivalence.

Chromium had exited before these controlled runs. Sprite was paused with an independently armed 80-second identity-checked resume watchdog and a 75-second controller deadline. Recorded pauses were approximately 23–43 seconds, with no process left stopped. Earlier uncontrolled thermal failures and all exploratory runs remain in the evidence directory.

## Reproduction and acceptance

```sh
python3 tools/build_wait_policy.py
python3 tools/build_launch_observer.py --resident
python3 tools/run_resident_decode.py --mode control --continuous --prime-tokens 2 \
  --cuda-wait blocking --model /path/to/Qwen3.5-2B-Q4_K_M.gguf \
  --ptx /path/to/gate-order-2b.ptx \
  --wait-library build/integration/libllama-cuda-wait.so \
  --output results/local-new-control
# Repeat with a new output directory and --mode mojo; keep every other setting equal.
```

All original AC and thermal guards remain. Desktop pausing is not performed by the public runner. Use a quiet comparison window and document desktop activity. The optional 40% CPU allowance was not tested because correcting wait policy was the better-controlled next experiment.

- Explicit wait targeting/readback: verified live in auto/blocking resident controls and the rebuilt public helper control.
- Capture separation: two-prime candidate trials recorded zero timed replacement/capture calls; no claim of counting graph replays.
- Workload validity: metadata regression tests and the reviewed fixture bound are in place; the actual fixture has 49 tokens.
- Verification: 75 CPU regression tests passed; worker/helper compiled under CPU/thermal guards. Focused reviews covered priming bounds, receipt parsing and controller ordering. A final receipt-sharing refactor was followed by the full suite.
- Whole-model Mojo speed/complete PDF quality: still unproved. Existing limited synthetic correctness checks do not satisfy this acceptance criterion.

[Raw evidence and derived comparison](../results/a2000/resident-wait-policy/comparison.json) include every successful comparison, earlier aborted retry, pause receipts and artifact hashes. The next comparison should cross over which independent context receives Mojo and check timing variability; a 2B operation benchmark would also establish whether this specialization has an isolated gain before expecting a whole-model benefit.
