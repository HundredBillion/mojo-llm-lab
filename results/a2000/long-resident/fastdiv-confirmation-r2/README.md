# Source-built fast-division candidate: confirmation attempt r2 (2026-11-07), INCOMPLETE

Protocol: [`docs/TSPs/11-07-2026-fast-division-confirmation.md`](../../../../docs/TSPs/11-07-2026-fast-division-confirmation.md) (committed in `c21bdc1` before measurements).
Candidate PTX SHA-256 `8e130a5a0a4e4355acbd0988bd31f627b09fdfcda31eedd7ce13fb05e52e9e3a`; worker `7446a8a4…`, observer `ef729b86…`, wait helper `b617385b…`, model `aaf42c8b…` (all re-hashed before the run).

This is the first performance batch for the corrected candidate. The prepared private batch (`fastdiv_confirmation_batch.py`) had never run; the `-r2` copies in `helpers/` differ only in status/stop-marker/output names, so the earlier session's stop marker and receipts were left untouched. Run directories were produced as `results/local-fastdiv-r2-confirmation-NN-…` and renamed here to the verifier's `confirmation-NN-…` layout; pause receipts were copied from the private state directory.

| Stage | Assignment | Status | CUDA/Mojo or control ratio | Peaks CPU/GPU/platform °C |
| --- | --- | --- | ---: | --- |
| 00 | control, arm 1 | passed | 0.99896574 | 61 / 78 / 54.05 |
| 01 | Mojo context 0 | passed | 1.02220388 | 57 / 81 / 56.05 |
| 02 | Mojo context 1 | passed | 1.01023753 | 82 / 80 / 57.05 |
| 03 | Mojo context 1 | **stopped at preflight**: CPU 90.0 °C (abort threshold), GPU 58 °C, AC on | — | 90 / 58 / 56.05 |
| 04 | Mojo context 0 | not run | — | — |
| 05 | control, arm 1 | not run | — | — |

The batch stopped as designed (`batch.json`: `stopped_after_failure`). No decision can be made: the rule needs four candidate and two control ratios, and `tools/verify_long_confirmation.py` will fail on the missing stages. **The batch was not retried in this session**; per protocol a retry must be a fresh, uniquely named full batch, not a fill-in of the missing stages, and must not be chosen based on the partial ratios above.

Completed stages all report the candidate PTX hash, top token 11140, two priming tokens and zero timed replacement/capture calls.

Safety notes:
- Stage 03's 90 °C reading was a transient spike in the 8 s gap after stage 02 (stage 02 peaked at 82 °C CPU). Package temperature was 54 °C shortly afterwards.
- Stage 01 reached GPU 81 °C, 1 °C below the 82 °C abort threshold.
- Fan session (`fan-session.json`): level 7, 30 s hardware watchdog, ended early with the stop marker, status `restored`, no errors, fan `level: auto`, watchdog restored to 0, `llm-fan.service` restarted. All paused Chromium/Sprite processes resumed (`still_stopped: []` in every receipt).
