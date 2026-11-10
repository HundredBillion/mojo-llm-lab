# Fast-division candidate confirmation, attempt r5 (2026-11-07): PASSED at a 1200 MHz GPU clock cap

Protocol: [base](../../../../docs/TSPs/11-07-2026-fast-division-confirmation.md), [r3 cool-down](../../../../docs/TSPs/11-07-2026-fast-division-confirmation-r3.md) and [r5 clock lock](../../../../docs/TSPs/11-07-2026-fast-division-confirmation-r5.md), committed in `de0a7f5` before measurement. Candidate PTX `8e130a5a0a4e4355acbd0988bd31f627b09fdfcda31eedd7ce13fb05e52e9e3a`.

| Stage | Assignment | Cool-down | Ratio (>1 favors Mojo) | Max SM clock | Peaks CPU/GPU/platform °C |
| --- | --- | ---: | ---: | ---: | --- |
| 00 | control | 7.1 s | 0.99683031 | 1200 | 52 / 79 / 56.05 |
| 01 | Mojo context 0 | 6.6 s | 1.02649667 | 1200 | 61 / 80 / 56.05 |
| 02 | Mojo context 1 | 5.6 s | 1.02339584 | 1200 | 51 / 80 / 57.05 |
| 03 | Mojo context 1 | 5.5 s | 1.01561332 | 1200 | 89 / 80 / 57.05 |
| 04 | Mojo context 0 | 5.3 s | 1.02473248 | 1200 | 63 / 80 / 57.05 |
| 05 | control | 5.8 s | 1.00195822 | 1200 | 87 / 80 / 57.05 |

Decision (`python3 tools/verify_long_confirmation.py results/a2000/long-resident/fastdiv-confirmation-r5`): all candidate ratios exceed one, log-margin **0.0123179672** against the required 0.005. **Passed.** The verifier also checked artifact identities, settings, context order, capture/replacement counters, zero timed host replacement/capture and top token 11140.

Scope and limits:
- Applies only to fixed-token (128) resident decode of Qwen3.5-2B Q4_K_M with the GPU graphics clock locked to 210–1200 MHz, on this laptop. It does not establish the advantage at default clocks, for prompt processing, sampled generation or PDF latency, or toward the 2× goal.
- Numerical qualification is the separate full-logit gate the same PTX passed earlier (exact equality, both contexts) at default clocks. Clock frequency does not change the instruction sequence, but r5 itself performed no logit validation.
- Safety margins remained thin: GPU peaked at 80 °C (guard 82), and CPU peaks reached 89 °C and 87 °C in stages 03 and 05 (guard 90).
- Fan/clock session restored with no errors: fan auto, watchdog 0, `llm-fan.service` active, persistence mode back to Disabled, clocks reset. All paused processes resumed.
