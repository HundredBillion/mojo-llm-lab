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
#define REPS 128
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
  for (int i = 0; i < 4096; i++) {
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
  LOAD(ggml_new_tensor_3d);
  LOAD(ggml_new_tensor_1d);
  LOAD(ggml_rms_norm);
  LOAD(ggml_mul);
  LOAD(ggml_rope_multi);
  LOAD(ggml_new_graph_custom);
  LOAD(ggml_build_forward_expand);
  LOAD(ggml_backend_cuda_init);
  LOAD(ggml_backend_alloc_ctx_tensors);
  LOAD(ggml_backend_tensor_set);
  LOAD(ggml_backend_tensor_get);
  LOAD(ggml_backend_graph_compute);
  LOAD(ggml_backend_buffer_free);
  LOAD(ggml_backend_free);
  float q[4096], gamma[128], reference[4096], output[4096];
  int32_t positions[4];
  FILE *in = fopen(argv[1], "rb");
  if (!in)
    return 4;
  int valid = fread(q, 4, 4096, in) == 4096 &&
              fread(gamma, 4, 128, in) == 128 &&
              fread(positions, 4, 4, in) == 4 && fgetc(in) == EOF;
  fclose(in);
  if (!valid)
    return 4;
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
  struct ggml_tensor *a = call_ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 32,
                                                  1),
                     *g = call_ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128),
                     *p = call_ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
  struct ggml_tensor *norm = call_ggml_rms_norm(ctx, a, 1e-6f),
                     *scaled = call_ggml_mul(ctx, norm, g);
  int sections[4] = {24, 20, 20, 0};
  struct ggml_tensor *result =
      call_ggml_rope_multi(ctx, scaled, p, NULL, 128, sections, 40, 32768,
                           5000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
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

  call_ggml_backend_tensor_set(a, q, 0, sizeof(q));
  call_ggml_backend_tensor_set(g, gamma, 0, sizeof(gamma));
  call_ggml_backend_tensor_set(p, positions, 0, sizeof(positions));
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
  CUdeviceptr out, theta;
  CK(cuMemAlloc(&out, sizeof(output) + 8));
  CK(cuMemAlloc(&theta, 4));
  uint32_t bits = 0x3f492c27;
  CK(cuMemcpyHtoD(theta, &bits, 4));
  const uint32_t sentinel = 0x7fc12345;
  CK(cuMemsetD32(out, sentinel, 4098));
  CK(cuCtxSynchronize());
  CUdeviceptr qp = (CUdeviceptr)a->data, gp = (CUdeviceptr)g->data,
              pp = (CUdeviceptr)p->data, op = out + 4;
  void *args[] = {&qp, &gp, &pp, &op, &theta};
  CUstream stream;
  CK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUgraph fused;
  CK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
  CK(cuLaunchKernel(function, 32, 1, 1, 128, 1, 1, 0, stream, args, NULL));
  CK(cuStreamEndCapture(stream, &fused));
  size_t baseline_kernels = inspect(captured), fused_kernels = inspect(fused);
  if (!baseline_kernels || fused_kernels != 1)
    return 11;
  struct timed_graph groups[2] = {repeated(captured), repeated(fused)};
  for (int i = 0; i < 2; i++) {
    CK(cuGraphUpload(groups[i].exec, stream));
    CK(cuStreamSynchronize(stream));
  }
  for (int j = 0; j < 24; j++)
    for (int i = 0; i < 2; i++) {
      CK(cuGraphLaunch(groups[i].exec, stream));
      CK(cuStreamSynchronize(stream));
    }
  CK(cuMemcpyDtoH(output, op, sizeof(output)));
  compare(output, reference, "warmup");
  printf("{\"repetitions_per_graph\":%d,\"baseline_kernel_nodes\":%zu,\"mojo_"
         "kernel_nodes\":%zu,\"context_flags\":%u,\"same_graph_control\":%s,"
         "\"samples\":[",
         REPS, baseline_kernels, fused_kernels, flags,
         control ? "true" : "false");
  int first = 1;
  const int abba[] = {0, 1, 1, 0};
  for (int round = 0; round < 8; round++)
    for (int slot = 0; slot < 4; slot++) {
      int mode = abba[slot], selected = control ? 0 : mode;
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
      unsigned mhz = 0;
      if (nvclock(nvdev, 0, &mhz))
        return 14;
      printf("%s{\"round\":%d,\"gpu_clock_mhz\":%u,\"mode\":\"%s\",\"gpu_us_"
             "per_operation\":%.9g,\"host_wall_us_per_operation\":%.9g,\"host_"
             "cpu_us_per_operation\":%.9g}",
             first ? "" : ",", round, mhz,
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
  CK(cuMemcpyDtoH(&edges[1], out + 4097 * 4, 4));
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
  CK(cuMemFree(theta));
  CK(cuModuleUnload(module));
  call_ggml_backend_buffer_free(buffer);
  call_ggml_backend_free(backend);
  call_ggml_free(ctx);
  return 0;
}
