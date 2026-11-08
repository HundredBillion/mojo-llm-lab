# Attempt r4 (25 W GPU power cap): not executed

No benchmark ran. The root session ran `nvidia-smi -pl 25`, which exited 0 but did not change the enforced limit (read back 35 W). The session refused to continue and restored everything within 0.1 s (`fan-window-status.json`: restored, no errors).

A separate root diagnostic (`gpu_cap_diagnostic.py` / `gpu-cap-diagnostic.json`) showed the driver prints "Changing power management limit is not supported for GPU" and only warns. Power limiting is unavailable on this laptop GPU. Clock locking (`-lgc 210,1200`) was accepted. Both were restored. Superseded by [r5](../../../../docs/TSPs/11-07-2026-fast-division-confirmation-r5.md).
