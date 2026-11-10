# Prompt processing (prefill), measured separately

Protocol: [`TSPs/11-07-2026-prefill-baseline.md`](TSPs/11-07-2026-prefill-baseline.md), committed in `3bf9b32` before measurement. Evidence: [`results/a2000/prefill-baseline/`](../results/a2000/prefill-baseline/).

## The Mojo candidate does not affect prefill

The qualified fast-division candidate replaces only the single-token Q4_K gate/up matrix-vector launch. In all six [coverage runs](../results/a2000/logit-coverage/), each candidate trial recorded only 55–104 replacements while prefilling 566–6772 tokens in 32-token batches. Replacement during prefill would have produced thousands: about 212 batches × 24 layers for 6772 tokens. **The confirmed decode gain (1.6–2.6 % at a 1200 MHz clock cap) therefore does not transfer to prompt processing.** Prompt processing has no Mojo speedup.

## CUDA prefill throughput at the reference server settings

Pinned llama-bench b12110 used full offload, Q4_0 K/V, flash attention, 2 threads, polling off, a 20 % CPU quota and verified blocking waits. Runs used default GPU clocks (max SM 1747–1762 MHz) and fans at level 7. `-n 0`: prefill only, 3 repetitions after llama-bench's warmup.

| Prompt tokens | batch / ubatch | Tokens/s (repetitions) | Mean | Peak CPU / GPU °C |
| ---: | --- | --- | ---: | --- |
| 512 | 32 / 32 | 1189.9, 1088.6, 1057.0 | **1111.9** | 75 / 65 |
| 2048 | 32 / 32 | 1159.7, 1160.2, 1164.4 | **1161.4** | 71 / 74 |
| 6144 | 32 / 32 | 1163.1, 1161.0, 1175.2 | **1166.4** | 87 / 80 |
| 2048 | 128 / 128 | 2425.9, 2636.3, 2860.7 | **2640.9** | 74 / 72 |
| 2048 | 512 / 512 | 3276.9, 3296.7, 3248.3 | **3274.0** | 81 / 71 |

At the reference setting (batch/ubatch 32), prefill runs at about 1.1–1.17 k tokens/s, roughly flat from 512 to 6144 tokens. A 6.8k-token prompt therefore needs roughly 6 s of prefill before generation starts.

**Batch size is the largest prefill lever observed, and it is not a Mojo change.** At 2048 tokens, ubatch 128 measured 2.27× and ubatch 512 2.82× the ubatch-32 throughput in llama-bench, with lower GPU peaks. Part of this is a llama-bench artifact. llama-bench computes the full 248,320-row vocabulary projection after **every** batch, which is 13.9 % of prefill GPU time at ubatch 32 (below). llama-server computes it only for the final prompt token, so a real server's gain from a larger ubatch is probably smaller. It must be measured in the server, together with VRAM headroom (4 GB, two-context or vision use), quality and complete PDF latency, before any server setting changes. No server settings were changed.

## Where prefill GPU time goes (ubatch 32, 2048 tokens)

The Nsight Systems capture recorded 116,424 kernel events and 2,247.8 ms of summed GPU duration, including llama-bench warmup. Profiler overhead invalidates its throughput.

| Operation | Calls | Share of summed GPU time |
| --- | ---: | ---: |
| Quantized matrix multiply (MMQ) Q4_K, incl. stream-k fixup | 15,744 | 33.99 % |
| MMQ Q5_K | 9,216 | 14.02 % |
| Vocabulary projection, Q6_K matrix-vector (llama-bench per-batch artifact) | 128 | 13.93 % |
| MMQ Q6_K | 3,840 | 7.54 % |
| Recurrent: gated delta net | 2,304 | 7.37 % |
| Activation quantization (q8_1 for MMQ) | 23,936 | 4.48 % |
| Recurrent: concat | 2,304 | 3.96 % |
| MMQ Q8_0 | 9,216 | 2.24 % |
| Flash attention | 1,536 | 2.22 % |
| Recurrent: ssm_conv | 2,304 | 2.15 % |

Prefill is dominated by **batched quantized matrix multiply** (MMQ, ~58 % across types). Decode, by contrast, is dominated by single-token matrix-vector kernels ([decode profile](qwen35-2b-profile.md)). A prefill-oriented Mojo kernel would need to beat llama.cpp's tuned MMQ, chiefly Q4_K at about 34 %, at 32-column tiles. That is a different and harder target than the decode replacement. Raising ubatch is a much cheaper candidate lever and should be tested in the real server first. Attention is a small share even at 2048 tokens; the hybrid model's recurrent layers (~13.5 %) are larger.

Recompute the aggregation without a GPU:

```sh
python3 tools/summarize_cuda_profile.py \
  results/a2000/prefill-baseline/prefill-05-p2048-ub32-nsys/kernel-events.jsonl.gz \
  --output /tmp/prefill-kernel-summary.json
```

The raw Nsight report and SQLite export stay local; their SHA-256 hashes are in `provenance.json`.

## Limits

These are synthetic llama-bench prompts on this laptop with these pinned artifacts. They do not measure PDF latency, vision encoding, sampled output or the server's scheduling. Throughput was measured at default clocks with level-7 fans. The 6144-token run reached GPU 80 °C and CPU 87 °C, so long prefills have little thermal margin. The session restored with no errors (fan auto, watchdog 0, `llm-fan.service` active).
