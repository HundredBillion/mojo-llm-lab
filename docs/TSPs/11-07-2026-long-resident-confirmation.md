# Independent confirmation after the exploratory long-trial batch

Prepared before any confirmation measurement. Do not pool this batch with earlier 16-token runs. Same worker, observer, PTX, model, wait helper, 128 timed tokens, two priming tokens, one ABBA cycle, CPU settings and cooling policy as the exploratory long batch.

Fixed schedule: CUDA control, Mojo context 0, context 1, context 1, context 0, CUDA control. Do not stop or retry based on the sign of a result. Stop on a safety or harness failure and retain the entire incomplete batch.

Decision rule for a limited fixed-token decode improvement: every confirmation candidate ratio must favor Mojo, and its smallest relative advantage must exceed the largest absolute log-ratio of the two CUDA/CUDA controls by at least 0.005 in log-ratio units. All required artifacts/settings must match, every run must pass, candidate timed capture/replacement counters must remain zero after priming, and fixed-prefix top tokens must match. Report the individual ratios and controls regardless of the decision. This rule is a conservative diagnostic, not a confidence interval or a guarantee against all correlated/systematic errors.

Still required separately: fuller output/logit quality checks, prompt-processing comparison, sampled generation and application/PDF validation. No 2× or general end-to-end claim follows from this experiment.
