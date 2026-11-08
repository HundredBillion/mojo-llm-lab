# Longer resident decode Technical Spec

> **For agentic workers:** Use dmi-superpowers:executing-plans for this sequential extension. Routine design and review gates are approved under workspace instructions.

**Goal:** Measure longer same-model decode trials without silently changing short-workload content or accepting stale workers.

**Architecture:** Extend the resident C++ fixture and capability handshake; validate capabilities in the Python controller before submitting work. Reuse the existing summarizer and thermal runner.

**Tech Stack:** Pinned llama.cpp C API, C++ worker, Python orchestration and unittest.

## Global Constraints

Keep existing AC/temperature guards, 512-token context, 2 CPU threads, blocking waits, 20% CPU allowance, two priming tokens, and the existing 128-thread candidate. No permanent host change. No whole-model or quality claim from microbenchmark evidence.

## Tasks

- [ ] Extend `integration/resident_decode.cpp`: repeat the synthetic text, verify its original first 42 token IDs, support up to 128 timed tokens, and emit `max_timed_tokens` in ready metadata.
- [ ] Extend `tools/run_resident_decode.py`: add explicit 128-token option; add `validate_ready` checking exact priming/assignment, fixture capacity and worker maximum; preserve verified capability metadata in the result.
- [ ] Add regression cases in `tests/test_resident_decode.py` for an old worker, short fixture, boundary length, missing capability fields and mismatched assignment/priming.
- [ ] Run focused/full CPU tests and compile the worker under CPU/thermal guards. Review the capability boundary and prefix preservation.
- [ ] Prepare the bounded fan session and fixed control, context-1, context-0, context-0, context-1, control schedule. Run only with verified cooling, keep failures, and recompute direction-aware ratios.
- [ ] Publish results and limitations to the existing PR; whole-model quality, prefill and end-to-end acceptance remain separate.
