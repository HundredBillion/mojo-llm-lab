# Resident host-cost Technical Spec

> Execute inline using executing-plans; routine approval stages are authorized by the workspace instructions.

Goal: make timed capture and CPU allowance visible without changing the default workload.
Architecture: snapshot existing atomic counters immediately around timed decode; validate deltas in the Python controller and retain quota metadata.
Tech stack: pinned llama.cpp C++ worker and Python controller/tests.

Constraints: CPU<90C, GPU<82C, platform<75C; AC required; existing cold gates. Quota limited to1000 or2000us per5000us, default1000. No permanent settings changes. Never pool mismatched worker artifacts.

- [x] Add timed counter fields to integration/resident_decode.cpp, outside the timed interval's clock boundaries.
- [x] Add --cpu-quota-us choices and explicit metadata in tools/run_resident_decode.py; validate timed/cumulative counter relationships.
- [x] Extend tests/test_resident_decode.py with missing, negative and inconsistent delta cases.
- [x] Run focused tests, guarded compilation and full regression tests.
- [x] Attempt bounded current-quota control/candidate instrumentation runs only after Sprite measurements finish; retain failed cold gates.
- [x] Report findings and remaining uncertainty before any higher-quota comparison.

Outcome: control passed; candidate attempt rejected before launch at CPU93C. Higher quota deferred. No new speedup claim or permanent change.


## Primed decode follow-up

- [x] Add optional priming worker argument and report it in handshake/trials.
- [x] Expose --prime-tokens, pass it to worker and validate matching workload metadata.
- [x] Add regression cases for priming mismatch/missing evidence; compile and run guarded tests.
- [x] Prepare identity-checked independent80-second resume watchdog for Chromium/Sprite; runner limited to75seconds and cleanup on timeout.
- [x] Run matched short controls/candidate only under original guards; retain raw outcomes and pause receipts.
- [x] Review results and publish bounded conclusions, without equating benchmark priming with a model optimization.

Two priming tokens are now selectable (default0, choices0/1/2). One left24 captures timed; two produced zero timed replacement/capture calls in the measured candidate. Validate the full prefill+prime+timed token count before each trial; the fixed Qwen fixture has49tokens. The timer excludes priming on both arms; this does not reduce actual request work.

Outcome: two priming tokens produced zero timed replacement captures; verified blocking waits cut CPU work in exploratory matched controls. Whole-model Mojo speedup remains unproved. Next separate work: cross over context assignments and qualify the isolated 2B operation.
