# Timing diagnostics and rejected controls

Clock observations and three additional runs did not reveal a Mojo speedup. One same-graph control was rejected, and that rejection is retained rather than rerun away.

| Configuration | Outcome | CUDA median | Mojo median |
| --- | --- | ---: | ---: |
| 32 repetitions, 2 warmup pairs, added clock sampling | Control rejected; no candidate run | — | — |
| 32 repetitions, 16 warmup pairs | Control passed | 113.792 µs | 128.175 µs |
| 256 repetitions, 16 warmup pairs | Control passed | 126.514 µs | 143.442 µs |

Raw records: [rejected control](../results/a2000/q4-clock-control-rejected/), [longer warmup](../results/a2000/q4-clock-warmup16/), [longer timing window](../results/a2000/q4-clock-long-window/).

## What the clocks establish

The harness now samples SM and memory clocks immediately before and after each timed graph, in addition to the existing graphics-clock sample. NVIDIA documents these as separate [NVML clock domains](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceEnums.html).

Memory samples were 5501 MHz throughout these runs. Longer GPU work produced varying SM/graphics samples, including 787–892 MHz in the long-window run. Point samples cannot rule out clock changes during a timed interval. They do not establish the cause of the remaining timing variability, and no clock setting was changed.

## Why pairing matters

In the rejected control, identical kernels shifted from approximately 96 to 116 microseconds midway through measurement. The two arms' marginal medians were 96.816 and 105.392 microseconds, outside the existing 5% control gate. That gate correctly stopped the candidate run under its declared rule.

A diagnostic that preserves ABBA round pairing gives a geometric-mean ratio of 0.988876. It computes each round's mean first/last baseline time divided by its mean middle candidate time, then combines all eight round ratios in log space. No samples or rounds are dropped. One transition round still differs by over 5%; pairing is not evidence of stationarity or a reason to override rejection.

The new diagnostic is supplemental. The original control gate remains unchanged. Tests distinguish true arm bias, linear drift, abrupt transitions, missing rounds, incorrect order and invalid times. Historical offline diagnostics are labeled separately from measurements written by the updated runner.

## No hotter default adopted

The long-window experiment peaked at 59°C CPU / 77°C GPU, versus at most 69°C GPU in the shorter warmup experiment. It did not eliminate variability. Defaults remain 32 repetitions and two warmup pairs; longer settings are explicit options:

```sh
python3 tools/build_q4_comparison.py --threads 128 --repetitions 256 --warmup-pairs 16
python3 tools/run_q4_cuda_comparison.py --variant packed --output results/local-long-window
```

Restore the default benchmark build with `python3 tools/build_q4_comparison.py --threads 128`. These are benchmark settings, not local LLM production defaults. Temporary desktop pauses stayed within the watchdog arrangement and were resumed; no permanent host setting changed.
