# Alternative-model trials on the A2000

Qwen3.5-2B Q4_K_M is the most promising next candidate for memory headroom on this 4 GB laptop. It passed three small synthetic invoice extractions, used approximately **1.6 GiB total device memory**, and measured **47.3 tokens/sec median** in four short decode bursts. Qwen3-4B-Instruct-2507 Q4_K_M passed the same chat-format cases, used about **2.7 GiB total device memory** in the measured balance case, and measured 36.8 tokens/sec with CUDA in the separate short comparison.

This is not a production-model selection or a proved PDF speed/cooling improvement. Models and tokenizers differ, the cross-model trials were sequential rather than interleaved, and the extraction times were mixed. The Mojo whole-model result remains inconclusive. Production defaults are unchanged.

## What was actually compared

| Model / execution | Small chat extraction cases | Short decode median | Scope |
| --- | --- | ---: | --- |
| Qwen3-4B-Instruct-2507 / CUDA | 3/3 correct | 36.8 tokens/sec | Four 16-token measurements |
| Qwen3-4B-Instruct-2507 / Mojo replacement | 3/3 correct; byte-identical to CUDA | 38.1 tokens/sec | Four 16-token measurements; no proved gain |
| Qwen3.5-2B / CUDA | 3/3 correct | 47.3 tokens/sec | Four 16-token measurements; no Mojo replacement |
| Existing Qwen3VL-4B / CUDA and Mojo | Basic chat case correct and byte-identical | Not measured in this comparison | One paired chat case only |

The three cases require a single JSON object with `Business` and `Total`. They cover a direct total, subtotal/tax plus an older paid invoice, and a total distinct from a remaining balance. These are synthetic text inputs, not real PDFs, OCR, images, transaction-table extraction, or comprehensive model-quality evaluation.

The Qwen3-4B model has the same 2560-wide / 9728-row feed-forward dimensions as the current kernel. A shadow run on its weights/activations compared **1119 calls / 10,885,632 values**, all within the unchanged tolerance, with maximum absolute difference 0.00000762939453. Its three chat outputs then remained byte-identical when the replacement fed forward.

The initial Qwen3.5-2B trial used explicit `--baseline-only`: its different dimensions (2048 / 6144) did not match the original kernel. That initial result is a model/runtime result, not a Mojo gain. A subsequent dedicated 2B specialization and same-model CUDA/Mojo comparison are recorded in [the Qwen3.5 integration report](qwen35-mojo.md).

## Prompt formatting changed the quality outcome

The unformatted completion prompt produced extra commentary even with CUDA on Qwen3-4B. It is therefore not a good adoption-quality test for this model, although the existing Qwen3VL raw-completion regression remains a reproducible numerical-sensitivity test.

The follow-up uses explicit system/user/assistant turns matching the models' documented chat formats. Qwen3.5 also receives its documented empty thinking block for non-thinking generation. The existing Qwen3VL model passes the basic formatted case with both implementations. That means a model switch alone cannot be credited for the quality improvement: prompt formatting is a material variable.

No tolerance was relaxed, the original failed output was not deleted, and no logit-equivalence claim is made. Three matching answers do not establish arbitrary-input equivalence.

## Whole-model Mojo timing is inconclusive

The attempted longer CUDA run (32 tokens × four repetitions) reached the unchanged GPU 82°C guard and was stopped. Its partial timings are excluded. The subsequent experiment used short, separately cold-started 16-token bursts in two CUDA–Mojo–Mojo–CUDA cycles, one measurement per process.

| Cycle | CUDA / Mojo mean decode-latency ratio | Direction |
| --- | ---: | --- |
| First | 1.0792× | Mojo faster |
| Second | 0.9691× | Mojo slower |

Pooled medians are 36.7874 versus 38.0525 tokens/sec, a 1.0344× ratio, but the cycles disagree and CUDA alone ranges from 35.55 to 41.55 tokens/sec. This does **not** establish a whole-model gain. Clock telemetry shows dynamic boost behavior; 100 ms device samples are not synchronized to the exact decode interval. The maximum sampled SM clocks ranged from 1665 to 1747 MHz. Short windows, startup/thermal state, and host scheduling remain confounders.

The [comparison summary and raw runs](../results/a2000/qwen3-instruct/decode-comparison.json) retain every sample. The isolated fused-operation evidence of 1.22–1.26× remains separate and unchanged.

## Smaller-model latency and memory

Logged inference times below include prompt processing, sampling and generation, but exclude model loading. Each is one run; none is a replicated task-latency claim.

| Synthetic extraction | Qwen3-4B CUDA | Qwen3.5-2B CUDA |
| --- | ---: | ---: |
| Basic invoice | 949.03 ms | 935.57 ms |
| Subtotal, tax, old paid invoice | 1069.62 ms | 1085.27 ms |
| Total versus remaining balance | 1150.52 ms | 1049.97 ms |

The 2B model's faster short decode does not translate into a comparably large improvement on these short extraction tasks. Different tokenization, prompt processing and sampling costs matter. The four 2B decode measurements range from 46.39 to 48.54 tokens/sec. Treat these as exploratory evidence for the smaller model, not a Mojo comparison or a PDF-service benchmark.

Sampled peak total device memory was 1652.75 MiB (1.614 GiB) for the 2B balance case, versus 2808.75 MiB (2.743 GiB) for the 4B balance case. These include desktop/device allocations, not solely model weights. No vision projector was loaded. Larger contexts and scanned-PDF processing require separate memory checks.

Short 2B decode runs peaked at GPU 71–72°C; the 4B short comparison peaked at 73–75°C. Sequential short trials with cooldowns do not prove sustained temperature or energy savings. One initial 4B download was stopped at CPU 90°C, and its slower retry passed. The 2B download also used a 5% CPU quota and reduced transfer rate. Sprite's independent watchdog resumed it after approximately 90 seconds during each long download; the complete download/hash jobs took longer than the pause.

## Reproduction and provenance

Both new files were downloaded at pinned quantizer revisions and fully SHA-256 verified before the model runs:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| Qwen3-4B-Instruct-2507-Q4_K_M.gguf | 2497281120 | `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` |
| Qwen3.5-2B-Q4_K_M.gguf | 1280835840 | `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223` |

The verified download records are separate from raw runner summaries, whose own hash field remains null. Source URLs, prompts, toolchain/artifact hashes, completions, temperatures, clocks, power and memory samples are retained under [qwen3-instruct](../results/a2000/qwen3-instruct/) and [qwen35-2b](../results/a2000/qwen35-2b/). Weights and binaries are not distributed.

All inference uses pinned llama.cpp b12110, full GPU-offload request, Q4_0 KV, flash attention, two CPU threads, polling disabled, batch/ubatch 32, a 20% CPU quota and the same blocking-wait shim. Completion contexts are 512 tokens; output caps are 32 tokens. These are bounded experimental settings, not document upload/length limits.

```sh
# Use exact prompts from the recorded summaries, with the appropriate model chat format.
MODEL_TEST_PROMPT=$(python3 -c 'import json; print(json.load(open("results/a2000/qwen3-instruct/qwen3-chat-cuda/summary.json"))["prompt"])')
python3 tools/run_launch_observer.py --model /path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --prompt "$MODEL_TEST_PROMPT" --tokens 32 --output results/local-quality-cuda
python3 tools/run_launch_observer.py --model /path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --prompt "$MODEL_TEST_PROMPT" --tokens 32 --replace-ptx /path/to/gate-order.ptx \
  --output results/local-quality-mojo
MODEL_TEST_PROMPT=$(python3 -c 'import json; print(json.load(open("results/a2000/qwen35-2b/chat/summary.json"))["prompt"])')
python3 tools/run_launch_observer.py --baseline-only --model /path/to/Qwen3.5-2B-Q4_K_M.gguf \
  --prompt "$MODEL_TEST_PROMPT" --tokens 32 --output results/local-quality-2b
# Short synthetic decode; omit --prompt. Repeat with fresh output directories and cooldowns.
python3 tools/run_launch_observer.py --baseline-only --model /path/to/model.gguf \
  --tokens 16 --repetitions 1 --output results/local-decode
```

Add `--wait-library /path/to/library.so` to reproduce the recorded local blocking-wait shim. The runner keeps AC/CPU/GPU/platform guards active. `--baseline-only` rejects validation/replacement flags, so a zero-match model cannot be mistaken for successful Mojo execution. Optional NVML telemetry records null if unavailable.

Official model configurations and templates: [Qwen3-4B config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/config.json), [Qwen3-4B chat template](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/tokenizer_config.json), [Qwen3.5-2B config](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json), [Qwen3.5-2B chat template](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/tokenizer_config.json).

Next useful work is a representative PDF/text-extraction quality suite for the 2B model, with complete task latency as the primary metric. Separately, the Mojo path needs matched-prefix logit diagnostics and a less variable model-timing method. None of these trials authorizes dropping quality checks or deploying the replacement by default.
