# First whole-model kernel attribution

A short synthetic decode of Qwen3VL-4B Q4_K_M on the RTX A2000 identified quantized matrix-vector work as the leading next optimization target. [Reviewed aggregate data](../results/a2000/decode-profile-summary.json) includes exact kernel names and counts.

| Family | Summed GPU kernel time | Share |
| --- | ---: | ---: |
| Q4_K/Q6_K matrix-vector | 311.028 ms | 87.84% |
| Attention | 14.563 ms | 4.11% |
| RMS normalization | 8.275 ms | 2.34% |
| RoPE | 2.968 ms | 0.84% |
| Other | 17.246 ms | 4.87% |

This is kernel-time attribution, not wall-clock attribution. It includes warmup and a single 16-token decode measurement starting at context depth zero. It does not establish long-context attention cost, prefill behavior, PDF accuracy, energy efficiency or a speedup. Profiling overhead and a 20% CPU quota also preclude using the reported application throughput as a production baseline.

The pinned b12110 engine used full GPU offload, Q4_0 KV, flash attention, two CPU threads, polling disabled and batch/ubatch 32. GPU clocks and fans remained automatic. Observed peaks were CPU 70°C and GPU 75°C; the temperature and AC guards stayed active. No permanent system settings changed.

## Capture recipe

With the pinned engine/model and Nsight Systems available in the benchmark environment, the capture invocation was:

```sh
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none --output=decode \
  llama-bench -m Qwen3VL-4B-Instruct-Q4_K_M.gguf \
  -p 0 -n 16 -r 1 -t 2 --poll 0 -ngl 99 \
  -b 32 -ub 32 -ctk q4_0 -ctv q4_0 -fa on -o json
nsys export --type sqlite --output decode.sqlite decode.nsys-rep
```

Run inside an externally monitored environment with the same guards as the kernel runner. This command alone does not provide temperature protection. The local container used blocking CUDA waits, a 20% CPU quota and writable user-owned output. An initial capture failed to save its trace because the output was unwritable; Nsight returned zero despite that failure. Always verify the report exists before accepting the run.

The SQLite aggregation joins `CUPTI_ACTIVITY_KIND_KERNEL.demangledName` to `StringIds.id`, sums `end - start`, and groups by kernel name. Type enum values 12 and 14 map to Q4_K and Q6_K in the pinned GGML header. The aggregate intentionally excludes local paths, complete profiler captures and private documents.

## Next experiment

Extract representative matrix shapes and quantization layouts from this model, then build a correctness-first Mojo Q4_K matrix-vector prototype against the real GGML operation. Preserve quantization blocks and input semantics. Compare against the already optimized CUDA kernel with matched buffers, timing controls and repeated measurements. Existing CPU Mojo K-quant code is a useful format reference, not a qualified GPU implementation.

Prioritize measured savings in the dominant operation over further polishing a small operation. Recheck kernel attribution at representative PDF context lengths before assuming this short-context distribution applies there. Apply the [whole-workload contract](benchmark-contract.md) before claiming user-visible gains.
