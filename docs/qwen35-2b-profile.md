# Current 2B model: measured CUDA kernel attribution

A guarded Nsight Systems capture of Qwen3.5-2B Q4_K_M completed on the RTX A2000 laptop with pinned llama.cpp b12110, full GPU offload, two CPU threads, polling disabled, Q4_0 KV, flash attention, batch/ubatch 32, and verified blocking CUDA waits. This is the CUDA baseline, without the Mojo replacement.

The trace contains 11,802 kernel events totaling 173.613353 ms of summed GPU duration. It includes warmup and the instrumented 16-token decode. The 17 large vocabulary-projection launches are consistent with that workload, but the trace has no explicit phase boundary used here; no warmup event was silently removed. Overlapping kernel durations remain summed, not converted into elapsed application time.

| Operation / launch group | Calls | GPU ms | Share of summed GPU duration |
| --- | ---: | ---: | ---: |
| Q6_K, grid 248320, unfused |17 |41.808496 |24.08% |
| Q4_K, grid 6144, fused gate/up target |408 |39.371623 |22.68% |
| Q5_K, grid 6144, unfused |306 |17.464204 |10.06% |
| Q6_K, grid 2048, fused |204 |13.076988 |7.53% |
| Q4_K, grid 2048, fused |289 |12.072235 |6.95% |
| Q5_K, grid 2048, fused |306 |8.322307 |4.79% |

Across all launch groups, Q4_K matrix-vector work totals 36.76%, Q6_K 31.91%, and Q5_K 14.85%. The existing Mojo specialization covers the 6144-row Q4_K gate/up group, not all Q4_K or all quantized matrix-vector work. Fused flags alone do not identify every tensor or fusion operation.

## Implications

The model header contains Q6_K token embeddings of dimensions [2048,248320] and no separate output tensor. The pinned model loader uses the token embedding weights for the output projection in that case. Thus the large matching Q6_K launch is consistent with the vocabulary projection. [Pinned model implementation](https://raw.githubusercontent.com/ggml-org/llama.cpp/cd959bae99ff4bed22bef2604f45241c09bbb8c8/src/models/qwen35.cpp)

Q6_K stores 256 weights in a 210-byte block. The corresponding weight tensor is 417,177,600 bytes. Dividing that size by the observed 2.459323 ms average launch gives approximately 169.6 GB/s of effective weight-read throughput, assuming one full tensor read per launch. This is an arithmetic estimate, not a hardware bandwidth counter or proof of saturation. [Pinned block layout](https://raw.githubusercontent.com/ggml-org/llama.cpp/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-common.h)

At the measured 22.68% GPU-time share, a hypothetical 1.14× gate/up improvement would ideally reduce summed GPU time by only about 2.8% if everything else stayed fixed. This does not predict application throughput: host work, profiling overhead, cache behavior and kernel interactions are excluded. It explains why the existing noisy whole-model comparison cannot establish the isolated gain's effect.

The next whole-model test should use the qualified 128-thread candidate, longer matched trials and repeatable cooling, retaining CUDA/CUDA controls and context assignment crossover. Further kernel candidates include the uncovered Q5_K/Q6_K operations, but they must beat the actual optimized CUDA implementation and preserve quantization and output correctness. There is no demonstrated 2× or smaller whole-model Mojo gain yet, nor any prompt-processing or PDF acceleration result from this trace.

## Cooling and experimental limits

The authenticated helper held normal maximum fan level 7 for three minutes with a 30-second hardware watchdog. Both fans reached roughly 5700/5100 RPM before GPU work. The profiler required both readings between 4500 and 10000 RPM throughout capture, rejecting the transient 65535 startup sentinel as invalid. CPU and GPU both peaked at 54°C; platform peak was 46.05°C. Sprite was paused for about 40 seconds with an independent resume watchdog and resumed successfully; Chromium was absent.

The helper restored automatic fan mode, watchdog 0 and the existing host fan service, with no restoration errors. This successful short window is not a controlled fan-only A/B experiment: starting temperatures and background conditions differed from earlier runs. It does not prove the cause of prior CPU spikes or sustained safety. No permanent cooling policy, Turbo setting or GPU clock was changed.

The first CPU-only SQLite export failed because Docker rejects a quota below 1000 µs. The retry used 1000 µs per 10000 µs, preserving a 10% allowance, and completed without GPU work. Both receipts are retained.

## Reproduction and evidence

Use the capture recipe in `docs/decode-profile.md` with this model and explicit `LLAMA_CUDA_WAIT_TARGET=/app/llama-bench`, `LLAMA_CUDA_WAIT=blocking`, and the built wait helper preloaded. Maintain external AC/thermal monitoring, verified fan behavior and the pinned artifacts. The exact container command, verified wait-policy receipt, sensor trace, fan restoration receipt and hashes are in [the evidence directory](../results/a2000/qwen35-2b-profile/provenance.json). Model SHA-256 was freshly checked under a CPU/thermal guard.

Recompute the public summary without a GPU:

```sh
python3 tools/summarize_cuda_profile.py \
  results/a2000/qwen35-2b-profile/kernel-events.jsonl.gz \
  --output /tmp/2b-kernel-summary.json
```

The event export contains every recorded kernel name, timestamp and launch geometry needed for the aggregation. Other Nsight metadata remains local; hashes of the original report and SQLite export are retained. The summarizer can also read a local Nsight SQLite export directly. It rejects unresolved names, invalid intervals, invalid dimensions and empty input. Tests cover overlapping kernel intervals and missing-name joins; focused read-only review independently confirmed the event count and summed duration. The profiler's reported tokens/sec is not accepted as a production baseline.
