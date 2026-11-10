# Fast-division candidate confirmation, attempt r3 (2026-11-07): INCOMPLETE, GPU thermal guard

Protocol: [base](../../../../docs/TSPs/11-07-2026-fast-division-confirmation.md) plus the [r3 cool-down amendment](../../../../docs/TSPs/11-07-2026-fast-division-confirmation-r3.md), committed in `deb4aa2` before measurement.

| Stage | Assignment | Cool-down wait | Status | Ratio | Peaks CPU/GPU/platform °C |
| --- | --- | ---: | --- | ---: | --- |
| 00 | control, arm 1 | 5.2 s | passed | 1.0008548347150767 | 77 / 81 / 57.05 |
| 01 | Mojo context 0 | 8.1 s | **aborted mid-run: "Temperature or AC guard"** (GPU reached 82 °C) | — | 87 / 82 / 58.05 |
| 02–05 | — | — | not run | — | — |

The cool-down passed both times. The abort happened during the measurement itself: the workload now drives the GPU to its 82 °C abort threshold at fan level 7. Across r2 and r3, GPU peaks per run were 78, 81, 80, 81 and 82 °C, so this protocol has essentially no thermal headroom on this laptop. No decision is possible. Not retried.

Fan session restored (auto, watchdog 0, `llm-fan.service` active, no errors); all paused processes resumed; no benchmark containers left running.
