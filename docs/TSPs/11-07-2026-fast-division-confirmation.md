# Confirm the source-built fast-division candidate

The original rounded-division candidate passed the timing rule but failed full-vocabulary validation. A single PTX-instruction diagnostic (`div.rn.f32` to `div.approx.ftz.f32`) eliminated every observed logit difference. That diagnostic is numerical evidence only. Implement the same change explicitly in a separate Mojo source variant; preserve the original source, artifacts and rejected results.

Before accepting the source-built candidate, validate its full logits with the unchanged tolerance and fixed 128-token fixture. Then run six new performance processes in this fixed order: CUDA control, Mojo context 0, Mojo context 1, Mojo context 1, Mojo context 0, CUDA control. Do not reuse the original candidate's speed results for the new candidate.

Retain the same worker, observer, model, wait library, two priming tokens, one ABBA cycle, 128 decode tokens, 20% CPU quota, blocking CUDA wait and thermal guards. All six new summaries must identify the new candidate PTX consistently. Run under a bounded maximum-normal-fan window, with independent desktop-resume watchdogs. Stop on safety/harness failure; keep incomplete batches and do not retry based on result sign.

Apply the original decision rule: all four candidate ratios must exceed one, and their smallest log-ratio minus the largest absolute log-ratio of the two controls must be at least 0.005. Require valid candidate capture/replacement counters, zero timed host replacement/capture, matching top tokens and consistent artifacts/settings. Numerical validation is a separate mandatory gate. This supports only fixed-token whole-model decode; prompt processing, sampled generation, PDF latency, sustained cooling and the 2× goal remain separate.
