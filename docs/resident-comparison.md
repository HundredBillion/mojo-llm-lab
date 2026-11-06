# Resident Qwen3.5-2B comparison and PDF smoke test

**A whole-model Mojo speedup remains unproved.** The completed short resident comparison showed a 1.0534× CUDA/Mojo latency ratio, but its matching CUDA/CUDA control showed a larger 1.0669× apparent difference. The new measurements expose significant CPU-quota throttling and a separate PDF correctness failure.

## What changed

The C++ worker loads one Qwen3.5-2B Q4_K_M model and creates two independent llama contexts. It serially selects CUDA or the opt-in Mojo projection replacement, clears model memory, prefills eight identical tokens, and times a fixed-token decode sequence through synchronization. Both contexts share a two-thread CPU pool with polling disabled. Graph caches are context-specific; changing the environment happens only between synchronized operations. Host replacement/capture counters verify candidate capture activity, not graph replay counts.

There are two warmup trials, followed by ABBA order. Warmup does **not** guarantee that memory clearing and the next prefill avoid later graph recaptures. This is teacher-forced whole-model decode, not sampled chat, PDF latency or a pure GPU-kernel measurement. All source uses the pinned b12110 API headers and binaries.

## Completed short comparison

Each arm has two measured 16-token trials in one ABBA cycle. The worker, observer, PTX, wait shim, model path/size and engine settings match between the short control and candidate runs. The same model is loaded once within each run. There is no cross-model speed attribution.

| Experiment | Arm 0 median | Arm 1 median | ABBA mean-latency ratio |
| --- | ---: | ---: | ---: |
| CUDA / CUDA control | 65.375 tokens/sec | 69.494 tokens/sec | 1.0669× |
| CUDA / Mojo candidate | 62.975 tokens/sec | 66.113 tokens/sec | 1.0534× |

A ~5% apparent candidate gain is not distinguishable from the ~6.7% same-implementation control difference here. Neither a gain nor a regression is established. The fixed-prefix top token matched across every completed short trial. This is a narrow numerical diagnostic, not comprehensive generated-output equivalence.

Each timed sequence records cgroup CPU usage, throttling count and throttled microseconds. The container is limited to 1 ms CPU time per 5 ms period (20% of one CPU). Reported throttling is approximately 80% of elapsed decode time in both completed short runs. This supports investigating host CPU scheduling/sampling costs; it is not a causal decomposition of latency. See the [kernel cgroup documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html#cpu-interface-files) for these counters.

The short runs peaked at CPU69/GPU71 for the control and CPU68/GPU73 for the candidate. Sprite was paused with a separate 90-second resume watchdog; runs lasted about 146 and 144 seconds, so the entire run was not quiet. No permanent CPU, Turbo, fan or GPU-clock setting changed.

## Retained earlier protocols and failures

- The initial 32-token CUDA/CUDA control completed two ABBA cycles with ratios 1.0014× and 0.9794×. Its saved temperature samples cover active receive phases only; cooldown safety checks ran but their readings were not retained. Later runs retain every sensor read.
- The matching 32-token candidate stopped during a cooldown at a sampled CPU93°C. Its partial timings are excluded from accepted comparisons.
- A continuous 32-token control block reached GPU82°C and was stopped. Its partial timings are also excluded.
- The final short protocol restored per-trial cooldowns and reduced the workload to 16 tokens. Different protocols are not pooled.

All [raw resident evidence and the derived comparison](../results/a2000/qwen35-resident/comparison.json) are retained. The original **4B isolated operation** result of 1.22–1.26× is unchanged; it is not a result for this 2B whole-model experiment.

## Reproduce and next investigation

```sh
python3 tools/fetch_upstream.py
python3 tools/build_launch_observer.py
python3 tools/build_launch_observer.py --resident
python3 tools/run_resident_decode.py --mode control --model /path/to/model.gguf \
  --ptx /path/to/gate-order-2b.ptx --wait-library /path/to/wait.so \
  --output results/local-control
# Repeat with --mode mojo and a fresh output directory.
python3 tools/summarize_resident.py \
  results/a2000/qwen35-resident/control-16-cooled/summary.json \
  results/a2000/qwen35-resident/mojo-16-cooled/summary.json
```

Compilation and inference must use the documented AC/temperature guards. Default resident settings are 16 tokens, one ABBA cycle and per-trial cooldowns. The explicit continuous option exists to reproduce the rejected protocol; it retains all absolute guards. Run CPU regression tests with `python3 -m unittest discover -s tests -q`.


The next useful work is to isolate CPU sampling/schema and graph-recapture costs, measure warmup/affinity effects, and qualify exact-decimal/date output contracts on representative documents. Raising CPU limits or adopting a format without a controlled thermal and quality comparison would not be justified by these results.


Verification completed: 70 CPU regression tests passed under the thermal guard, including invalid evidence and cleanup-after-daemon-failure cases. The observer and resident C++ worker compiled successfully in the pinned CPU-limited container. The existing operation-evidence verifier still finds all 24 paired rounds favoring the 4B Mojo kernel. Focused code and evidence reviews found no remaining blocking issue; no full PDF capability or whole-model gain is inferred from these checks.
