# Context crossover Technical Spec

Execute inline under the workspace's approved workflow preference.

- [x] Add worker candidate-context argument0|1, default1; preserve fixed assignment for each context lifetime and report metadata.
- [x] Add --mojo-arm to controller, validate handshake/trial metadata and apply CUDA/candidate counter checks to the selected arm.
- [x] Add directed candidate ratios and workload validation to summarizer; regression tests must catch a reversed assignment becoming a false speedup.
- [x] Run guarded C++ build and full CPU suite, review scope, then planned control/1/0/0/1/control batch using existing verified blocking helper.
- [x] Preserve outcomes and independently verify ratios/identity/cleanup. Publish conclusions without treating tests or one positive sample as whole-model proof.

All existing AC/temperature guards and20% CPU limit remain. Use priming2,16timed tokens and one ABBA cycle. One process per experiment; no graph-policy switch within a context. Candidate assignment1,0,0,1 is fixed before measurement. Stop batch on a guard failure.

Outcome: all six planned runs completed. Candidate geometric-mean ratio0.98979 provides no whole-model speedup proof. Next investigate isolated2B operation performance.
