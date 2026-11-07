# Qualify the 2B gate/up kernel

Determine whether the existing Mojo fused gate/up kernel improves the isolated operation used by Qwen3.5-2B before further whole-model tuning. Use 2048 inputs and 6144 distinct output rows, identical Q4_K weights and Q8_1 activation quantization, pinned llama.cpp CUDA, correctness and red-zone checks, ABBA measurements and a same-CUDA-graph control. Preserve all existing 4B fixture bytes and evidence.

Extend the existing harness with explicit dimensions rather than copying it or changing model architecture. Reject incompatible fixtures before GPU work. Record dimensions in native output and verify them. Existing power, temperature, and duration guards remain mandatory. Retain failed attempts. An isolated win does not establish whole-model improvement.

Review: model dimensions are fixed; launch geometry may change later. Initial qualification uses the existing 128-thread PTX. Future 64/256-thread experiments require separate correctness qualification. Synthetic weights test the operation, not model quality. No permanent server setting changes are justified yet.
