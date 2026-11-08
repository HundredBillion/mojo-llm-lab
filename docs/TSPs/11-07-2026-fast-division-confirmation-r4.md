# Amendment r4: temporary 25 W GPU power cap for the fast-division confirmation

Amends the [base protocol](11-07-2026-fast-division-confirmation.md) and the [r3 cool-down amendment](11-07-2026-fast-division-confirmation-r3.md). Committed before any r4 measurement.

r3 stopped mid-run when the GPU reached the 82 °C guard. Across r2 and r3, per-run GPU peaks were 78–82 °C. During active phases, the GPU sat at its 35 W default power limit (P95 ≈ 35 W). Repeating the identical protocol is likely to abort again.

**Change:** for the whole r4 batch, the GPU power limit is lowered from 35 W to **25 W** (`nvidia-smi -pl 25`), and nothing else changes. The cap is applied (and read back via `enforced.power.limit`) inside the existing root, bounded fan session after the fans reach level 7. It is read back and must equal 25 W. The original limit, required to be 35 W or the session refuses to start, is restored first in the session's cleanup on every exit path, read back and recorded. A reboot also resets the limit. Before every stage the batch re-reads the limit and stops if it is not 25 W, and it records the value per stage. The r3 cool-down wait is retained.

**Scope of any result:** the cap applies to the CUDA control and the Mojo candidate alike, so the paired comparison stays matched. A pass, however, establishes a fixed-token decode advantage **only at a 25 W GPU limit**. It does not transfer to the default 35 W configuration. All other settings, artifacts, the schedule, the decision rule and the separate numerical gate are unchanged. A stop on a safety or harness failure is kept as an incomplete result and is not retried on the basis of partial ratios.
