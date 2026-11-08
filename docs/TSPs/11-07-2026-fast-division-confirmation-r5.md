# Amendment r5: temporary GPU clock lock (≤ 1200 MHz) for the fast-division confirmation

Supersedes the [r4 power-cap amendment](11-07-2026-fast-division-confirmation-r4.md), which could not be applied: power limiting is unsupported on this GPU ([r4 record](../../results/a2000/long-resident/fastdiv-confirmation-r4/)). Retains the [r3 cool-down](11-07-2026-fast-division-confirmation-r3.md). Committed before any r5 measurement.

**Change:** for the whole batch, the GPU graphics clock is locked to 210–1200 MHz (`nvidia-smi -lgc 210,1200`), and persistence mode is enabled temporarily so the lock survives between processes. Earlier runs had median SM clocks of about 950–1200 MHz and 1600–1700 MHz boost peaks, with GPU peaks of 78–82 °C. Both changes are applied inside the bounded root fan session after the fans reach level 7, and the session refuses to start unless both succeed. On every exit path they are reverted (`-rgc`, original persistence mode) and read back. A reboot also clears both.

**Verification:** the lock has no direct readback, so each stage is validated from its own recorded `sm_clock_MHz` samples. If any sample exceeds 1215 MHz, the stage is invalid and the batch stops.

**Scope:** the lock applies equally to the CUDA control and the Mojo candidate. A pass establishes a fixed-token decode advantage **only with the GPU clock capped at 1200 MHz**, not at default clocks. All other settings, artifacts, the schedule, the decision rule and the numerical gate are unchanged. Safety or harness stops are kept as incomplete and are not retried on the basis of partial ratios.
