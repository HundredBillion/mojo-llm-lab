# Resident same-model comparison

Load the pinned Qwen3.5-2B model once and give two independent llama contexts identical fixed token inputs. Warm each context with its assigned implementation before timing. Alternate ABBA trials, reset model memory between trials, and synchronize before stopping the clock. CPU throttling must be measured over the timed interval. Run a CUDA/CUDA control before CUDA/Mojo. Preserve every result and the existing AC and temperature guards; stop rather than relaxing them.

Separate graph caches are essential: changing the replacement environment cannot change an already captured graph. Only one context executes at a time, and environment changes happen after synchronization. Replacement/capture counters provide warmup evidence, not graph replay counts. The model and contexts stay loaded during controller-directed cooldowns.

This teacher-forced decode test excludes token sampling and document processing. It cannot establish PDF latency or arbitrary output equivalence. A subsequent synthetic PDF check must exercise a real application's text extraction/parser seam without mutating live user documents. Image-only extraction remains a separate capability.

Alternatives considered: more process-level trials retain startup confounding; fixed GPU clocks require system changes and do not isolate host scheduling. The resident harness reduces one source of variance without changing hardware policy. It still needs a same-implementation control and honest treatment of residual variation.


## Revised bounded-block protocol

The initial per-trial cooldown control completed, but a candidate run hit CPU 93°C during a cooldown and was removed. Loaded idle contexts also kept the GPU drawing power. Preserve both runs as diagnostics. For subsequent runs, use the same initial cold gate and a post-load cold gate, then execute the two warmups and eight timed trials as one short ABBA block. Continue monitoring every trial against the unchanged CPU90/GPU82/platform75/AC limits. Use the previously authorized Sprite pause with an independent 90-second resume watchdog. Apply this revised protocol to both control and candidate; do not pool it with the earlier cooldown protocol. No physical temperature limit changes.


## Short cooled protocol

The continuous 32-token control hit GPU82 and was rejected. Use 16 tokens and one ABBA cycle with per-trial cold gates for both comparison arms, still with the bounded Sprite pause. Retain the longer control and rejected runs separately; never pool different protocols. A completed control is required to interpret the matching candidate. Initial control data show substantial CPU-quota throttling and nonzero same-implementation variance, so a small positive ratio cannot establish a Mojo gain. Context warmup does not guarantee that memory clearing and prefill avoid all later graph recaptures; host counters include preparation and capture, not replay counts.
