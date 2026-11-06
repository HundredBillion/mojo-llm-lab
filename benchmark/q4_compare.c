#define _GNU_SOURCE
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(x)                                                                  \
  do {                                                                         \
    CUresult rc = (x);                                                         \
    if (rc != CUDA_SUCCESS) {                                                  \
      const char *s = NULL;                                                    \
      cuGetErrorString(rc, &s);                                                \
      fprintf(stderr, "%s: %d %s\n", #x, rc, s ? s : "");                      \
      exit(20);                                                                \
    }                                                                          \
  } while (0)
#define LOAD(name)                                                             \
  __typeof__(&name) call_##name =                                              \
      (__typeof__(&name))dlsym(RTLD_DEFAULT, #name);                           \
  if (!call_##name) {                                                          \
    fprintf(stderr, "missing %s\n", #name);                                    \
    return 3;                                                                  \
  }
#ifndef REPS
#define REPS 32
#endif
#ifndef WARMUP_PAIRS
#define WARMUP_PAIRS 2
#endif
#define ROWS 9728
#ifndef MOJO_THREADS
#define MOJO_THREADS 32
#endif
static CUgraph captured;
static int armed, capture_error, captures;

/* Clone only the GGML capture explicitly armed by this standalone benchmark. */
cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream,
                                           cudaGraph_t *graph) {
  static cudaError_t (*real_call)(cudaStream_t, cudaGraph_t *);
  if (!real_call)
    real_call = dlsym(RTLD_NEXT, "cudaStreamEndCapture");
  if (!real_call) {
    fprintf(stderr, "Missing real capture function\n");
    exit(21);
  }
  cudaError_t rc = real_call(stream, graph);
  if (armed && rc == cudaSuccess && !captured) {
    capture_error = cuGraphClone(&captured, (CUgraph)*graph);
    captures++;
  }
  return rc;
}

static double clock_seconds(clockid_t id) {
  struct timespec t;
  if (clock_gettime(id, &t))
    exit(22);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static size_t inspect(CUgraph graph) {
  size_t n = 0;
  CK(cuGraphGetNodes(graph, NULL, &n));
  CUgraphNode *nodes = calloc(n, sizeof(*nodes));
  if (!nodes)
    exit(23);
  CK(cuGraphGetNodes(graph, nodes, &n));
  size_t kernels = 0;
  for (size_t i = 0; i < n; i++) {
    CUgraphNodeType type;
    CK(cuGraphNodeGetType(nodes[i], &type));
    if (type == CU_GRAPH_NODE_TYPE_KERNEL)
      kernels++;
    else if (type == CU_GRAPH_NODE_TYPE_GRAPH) {
      CUgraph child;
      CK(cuGraphChildGraphNodeGetGraph(nodes[i], &child));
      kernels += inspect(child);
    } else if (type != CU_GRAPH_NODE_TYPE_EMPTY) {
      fprintf(stderr, "Unexpected leaf type %d\n", type);
      exit(24);
    }
  }
  free(nodes);
  return kernels;
}
struct timed_graph {
  CUgraph graph;
  CUgraphExec exec;
  CUevent start, stop;
};
static struct timed_graph repeated(CUgraph child) {
  struct timed_graph t = {0};
  CUgraphNode last, next;
  CK(cuGraphCreate(&t.graph, 0));
  CK(cuEventCreate(&t.start, CU_EVENT_BLOCKING_SYNC));
  CK(cuEventCreate(&t.stop, CU_EVENT_BLOCKING_SYNC));
  CK(cuGraphAddEventRecordNode(&last, t.graph, NULL, 0, t.start));
  for (int i = 0; i < REPS; i++) {
    CK(cuGraphAddChildGraphNode(&next, t.graph, &last, 1, child));
    last = next;
  }
  CK(cuGraphAddEventRecordNode(&next, t.graph, &last, 1, t.stop));
  CK(cuGraphInstantiateWithFlags(&t.exec, t.graph, 0));
  return t;
}
static void destroy(struct timed_graph t) {
  CK(cuGraphExecDestroy(t.exec));
  CK(cuGraphDestroy(t.graph));
  CK(cuEventDestroy(t.start));
  CK(cuEventDestroy(t.stop));
}
static void compare(const float *actual, const float *expected,
                    const char *label) {
  double largest = 0;
  int bits = 0;
  for (int i = 0; i < ROWS; i++) {
    double error = fabs((double)actual[i] - expected[i]);
    if (error > largest)
      largest = error;
    bits += memcmp(&actual[i], &expected[i], 4) != 0;
    if (!isfinite(actual[i]) || !isfinite(expected[i]) ||
        error > 1e-5 + 1e-5 * fabs(expected[i])) {
      fprintf(stderr, "Incorrect %s index%d actual%.9g expected%.9g\n", label,
              i, actual[i], expected[i]);
      exit(25);
    }
  }
  fprintf(stderr, "correct %s max_abs=%.9g bit_mismatches=%d\n", label, largest,
          bits);
}
int main(int argc, char **argv) {
  int control = argc == 5 && strcmp(argv[4], "--same-graph-control") == 0;
  if (argc != 4 && !control) {
    fprintf(stderr, "usage: bench INPUT PTX ENTRY [--same-graph-control]\n");
    return 2;
  }
  CK(cuInit(0));
  CUdevice device;
  CK(cuDeviceGet(&device, 0));
  unsigned initial_flags;
  int active;
  CK(cuDevicePrimaryCtxGetState(device, &initial_flags, &active));
  if (active) {
    fprintf(stderr, "Standalone benchmark requires inactive primary context\n");
    return 15;
  }
  CK(cuDevicePrimaryCtxSetFlags(device, (initial_flags & ~7u) |
                                            CU_CTX_SCHED_BLOCKING_SYNC));
  if (!dlopen("/app/libggml.so", RTLD_NOW | RTLD_GLOBAL) ||
      !dlopen("/app/libggml-cuda.so", RTLD_NOW | RTLD_GLOBAL)) {
    fprintf(stderr, "%s\n", dlerror());
    return 3;
  }
  LOAD(ggml_init);
  LOAD(ggml_free);
  LOAD(ggml_new_tensor_2d);
  LOAD(ggml_mul_mat);
  LOAD(ggml_new_graph_custom);
  LOAD(ggml_build_forward_expand);
  LOAD(ggml_backend_cuda_init);
  LOAD(ggml_backend_alloc_ctx_tensors);
  LOAD(ggml_backend_tensor_set);
  LOAD(ggml_backend_tensor_get);
  LOAD(ggml_backend_graph_compute);
  LOAD(ggml_backend_buffer_free);
  LOAD(ggml_backend_free);
  static unsigned char weights[ROWS * 1440], activation[2880];
  float input[2560], reference[ROWS], output[ROWS];
  char path[4096];
  snprintf(path, sizeof(path), "%s/weights.bin", argv[1]);
  FILE *in = fopen(path, "rb");
  if (!in)
    return 4;
  int valid = fread(weights, 1, 64 * 1440, in) == 64 * 1440 && fgetc(in) == EOF;
  fclose(in);
  if (!valid)
    return 4;
  for (int tile = 1; tile < 152; tile++)
    memcpy(weights + tile * 64 * 1440, weights, 64 * 1440);
  snprintf(path, sizeof(path), "%s/activations.bin", argv[1]);
  in = fopen(path, "rb");
  if (!in)
    return 4;
  valid = fread(activation, 1, sizeof(activation), in) == sizeof(activation) &&
          fgetc(in) == EOF;
  fclose(in);
  if (!valid)
    return 4;
  for (int group = 0; group < 80; group++) {
    uint16_t delta;
    memcpy(&delta, activation + group * 36, 2);
    if (delta != 0x2000)
      return 4;
    for (int j = 0; j < 32; j++)
      input[group * 32 + j] =
          (float)(int8_t)activation[group * 36 + 4 + j] / 128.0f;
    if (input[group * 32] != 127.0f / 128.0f)
      return 4;
  }
  struct ggml_init_params params = {8 * 1024 * 1024, NULL, true};
  struct ggml_context *ctx = call_ggml_init(params);
  if (!ctx)
    return 5;
  ggml_backend_t backend = call_ggml_backend_cuda_init(0);
  if (!backend)
    return 6;
  void *nv = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!nv)
    return 14;
  int (*nvinit)(void) = dlsym(nv, "nvmlInit_v2");
  int (*nvget)(const char *, void **) =
      dlsym(nv, "nvmlDeviceGetHandleByPciBusId_v2");
  int (*nvclock)(void *, unsigned, unsigned *) =
      dlsym(nv, "nvmlDeviceGetClockInfo");
  char pci[32];
  CK(cuDeviceGetPCIBusId(pci, sizeof(pci), device));
  void *nvdev = NULL;
  if (!nvinit || !nvget || !nvclock || nvinit() || nvget(pci, &nvdev))
    return 14;
  struct ggml_tensor *a =
      call_ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2560, ROWS);
  struct ggml_tensor *b = call_ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
  struct ggml_tensor *result = call_ggml_mul_mat(ctx, a, b);
  struct ggml_cgraph *graph = call_ggml_new_graph_custom(ctx, 64, false);
  call_ggml_build_forward_expand(graph, result);
  ggml_backend_buffer_t buffer =
      call_ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (!buffer)
    return 8;
  unsigned flags = 0;
  CK(cuCtxGetFlags(&flags));
  if ((flags & 7) != CU_CTX_SCHED_BLOCKING_SYNC) {
    fprintf(stderr, "Context not blocking: %u\n", flags);
    return 7;
  }

  call_ggml_backend_tensor_set(a, weights, 0, sizeof(weights));
  call_ggml_backend_tensor_set(b, input, 0, sizeof(input));
  armed = 1;
  for (int i = 0; i < 4; i++)
    if (call_ggml_backend_graph_compute(backend, graph))
      return 9;
  armed = 0;
  if (!captured || capture_error || captures != 1) {
    fprintf(stderr, "No valid GGML CUDA graph captured (%d/%d)\n", captures,
            capture_error);
    return 10;
  }
  call_ggml_backend_tensor_get(result, reference, 0, sizeof(reference));
  CUmodule module;
  CUfunction function;
  CK(cuModuleLoad(&module, argv[2]));
  CK(cuModuleGetFunction(&function, module, argv[3]));
  size_t count = 0;
  CK(cuGraphGetNodes(captured, NULL, &count));
  if (count != 2) {
    fprintf(stderr, "Expected quantize and matvec nodes, got %zu\n", count);
    return 16;
  }
  CUgraphNode nodes[2];
  CK(cuGraphGetNodes(captured, nodes, &count));
  CUDA_KERNEL_NODE_PARAMS quant = {0};
  int mojo_registers = 0, baseline_registers = 0, mojo_local = 0,
      baseline_local = 0;
  int mojo_blocks = 0, baseline_blocks = 0;
  CK(cuFuncGetAttribute(&mojo_registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function));
  CK(cuFuncGetAttribute(&mojo_local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                        function));
  CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&mojo_blocks, function,
                                                 MOJO_THREADS, 0));
  int found = 0;
  for (size_t i = 0; i < count; i++) {
    CUgraphNodeType type;
    CK(cuGraphNodeGetType(nodes[i], &type));
    if (type != CU_GRAPH_NODE_TYPE_KERNEL)
      return 16;
    CUDA_KERNEL_NODE_PARAMS params;
    CK(cuGraphKernelNodeGetParams(nodes[i], &params));
    const char *name = NULL;
    CK(cuFuncGetName(&name, params.func));
    if (strstr(name, "quantize_q8_1")) {
      quant = params;
      found++;
    } else {
      CK(cuFuncGetAttribute(&baseline_registers, CU_FUNC_ATTRIBUTE_NUM_REGS,
                            params.func));
      CK(cuFuncGetAttribute(&baseline_local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                            params.func));
      CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(
          &baseline_blocks, params.func,
          params.blockDimX * params.blockDimY * params.blockDimZ,
          params.sharedMemBytes));
    }
  }
  if (found != 1 || !quant.kernelParams)
    return 16;
  CUdeviceptr ap = *(CUdeviceptr *)quant.kernelParams[1];
  unsigned char actual_activation[2880];
  CK(cuMemcpyDtoH(actual_activation, ap, sizeof(actual_activation)));
  if (memcmp(activation, actual_activation, sizeof(activation))) {
    fprintf(stderr, "Activation bytes differ\n");
    return 17;
  }
  CUdeviceptr out;
  CK(cuMemAlloc(&out, sizeof(output) + 8));
  const uint32_t sentinel = 0x7fc12345;
  CK(cuMemsetD32(out, sentinel, ROWS + 2));
  CK(cuCtxSynchronize());
  CUdeviceptr wp = (CUdeviceptr)a->data, op = out + 4;
  CUdeviceptr metadata_device = 0, quants_device = 0;
#ifdef PREPACKED
  static unsigned char metadata[ROWS * 160], quants[ROWS * 1280];
  for (int i = 0; i < ROWS * 10; i++) {
    memcpy(metadata + i * 16, weights + i * 144, 16);
    memcpy(quants + i * 128, weights + i * 144 + 16, 128);
  }
  CK(cuMemAlloc(&metadata_device, sizeof(metadata)));
  CK(cuMemAlloc(&quants_device, sizeof(quants)));
  CK(cuMemcpyHtoD(metadata_device, metadata, sizeof(metadata)));
  CK(cuMemcpyHtoD(quants_device, quants, sizeof(quants)));
  void *args[] = {&metadata_device, &quants_device, &ap, &op};
  (void)wp;
#else
  void *args[] = {&wp, &ap, &op};
#endif
  CK(cuCtxSynchronize());
  CUstream stream;
  CK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUgraph fused;
  CK(cuGraphCreate(&fused, 0));
  CUgraphNode qnode, mnode;
  CK(cuGraphAddKernelNode(&qnode, fused, NULL, 0, &quant));
  CUDA_KERNEL_NODE_PARAMS mojo = {0};
  mojo.func = function;
  mojo.gridDimX = ROWS;
  mojo.gridDimY = mojo.gridDimZ = 1;
  mojo.blockDimX = MOJO_THREADS;
  mojo.blockDimY = mojo.blockDimZ = 1;
  mojo.kernelParams = args;
  CK(cuGraphAddKernelNode(&mnode, fused, &qnode, 1, &mojo));
  size_t baseline_kernels = inspect(captured), fused_kernels = inspect(fused);
  if (!baseline_kernels || fused_kernels != 2)
    return 11;
  struct timed_graph groups[2] = {repeated(captured), repeated(fused)};
  for (int i = 0; i < 2; i++) {
    CK(cuGraphUpload(groups[i].exec, stream));
    CK(cuStreamSynchronize(stream));
  }
  for (int j = 0; j < WARMUP_PAIRS; j++)
    for (int i = 0; i < 2; i++) {
      CK(cuGraphLaunch(groups[i].exec, stream));
      CK(cuStreamSynchronize(stream));
    }
  CK(cuMemcpyDtoH(output, op, sizeof(output)));
  compare(output, reference, "warmup");
  printf("{\"mojo_registers\":%d,\"baseline_registers\":%d,\"mojo_local_"
         "bytes\":%d,\"baseline_local_bytes\":%d,\"mojo_max_blocks_per_sm\":%d,"
         "\"baseline_max_blocks_per_sm\":%d,\"warmup_pairs\":%d,\"repetitions_"
         "per_graph\":%d,\"baseline_kernel_"
         "nodes\":%zu,\"mojo_"
         "kernel_nodes\":%zu,\"context_flags\":%u,\"same_graph_control\":%s,"
         "\"samples\":[",
         mojo_registers, baseline_registers, mojo_local, baseline_local,
         mojo_blocks, baseline_blocks, WARMUP_PAIRS, REPS, baseline_kernels,
         fused_kernels, flags, control ? "true" : "false");
  int first = 1;
  const int abba[] = {0, 1, 1, 0};
  for (int round = 0; round < 8; round++)
    for (int slot = 0; slot < 4; slot++) {
      int mode = abba[slot], selected = control ? 0 : mode;
      unsigned memory_before = 0, sm_before = 0;
      if (nvclock(nvdev, 2, &memory_before) || nvclock(nvdev, 1, &sm_before))
        return 14;
      double wall = clock_seconds(CLOCK_MONOTONIC),
             cpu = clock_seconds(CLOCK_PROCESS_CPUTIME_ID);
      CK(cuGraphLaunch(groups[selected].exec, stream));
      CK(cuStreamSynchronize(stream));
      cpu = clock_seconds(CLOCK_PROCESS_CPUTIME_ID) - cpu;
      wall = clock_seconds(CLOCK_MONOTONIC) - wall;
      float ms;
      CK(cuEventElapsedTime(&ms, groups[selected].start,
                            groups[selected].stop));
      if (!(ms > 0) || !isfinite(ms))
        return 12;
      unsigned mhz = 0, memory_after = 0, sm_after = 0;
      if (nvclock(nvdev, 0, &mhz) || nvclock(nvdev, 2, &memory_after) ||
          nvclock(nvdev, 1, &sm_after))
        return 14;
      printf("%s{\"round\":%d,\"gpu_clock_mhz\":%u,\"memory_clock_before_mhz\":"
             "%u,\"memory_clock_after_mhz\":%u,\"sm_clock_before_mhz\":%u,\"sm_"
             "clock_after_mhz\":%u,\"mode\":\"%s\",\"gpu_us_"
             "per_operation\":%.9g,\"host_wall_us_per_operation\":%.9g,\"host_"
             "cpu_us_per_operation\":%.9g}",
             first ? "" : ",", round, mhz, memory_before, memory_after,
             sm_before, sm_after,
             mode ? (control ? "ggml_control" : "mojo") : "ggml",
             ms * 1000 / REPS, wall * 1e6 / REPS, cpu * 1e6 / REPS);
      first = 0;
    }
  CK(cuMemcpyDtoH(output, op, sizeof(output)));
  compare(output, reference, "timed Mojo");
  call_ggml_backend_tensor_get(result, output, 0, sizeof(output));
  compare(output, reference, "timed GGML");
  uint32_t edges[2];
  CK(cuMemcpyDtoH(&edges[0], out, 4));
  CK(cuMemcpyDtoH(&edges[1], out + (ROWS + 1) * 4, 4));
  if (edges[0] != sentinel || edges[1] != sentinel)
    return 13;
  printf("],\"correct\":true,\"red_zones\":\"intact\"}\n");
  for (int i = 0; i < 2; i++) {
    destroy(groups[i]);
  }
  CK(cuGraphDestroy(captured));
  CK(cuGraphDestroy(fused));
  CK(cuStreamDestroy(stream));
  CK(cuMemFree(out));
  if (metadata_device)
    CK(cuMemFree(metadata_device));
  if (quants_device)
    CK(cuMemFree(quants_device));
  CK(cuModuleUnload(module));
  call_ggml_backend_buffer_free(buffer);
  call_ggml_backend_free(backend);
  call_ggml_free(ctx);
  return 0;
}
