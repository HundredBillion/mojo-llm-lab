#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include <cuda.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define LOAD(name)                                                             \
  __typeof__(&name) call_##name =                                              \
      (__typeof__(&name))dlsym(RTLD_DEFAULT, #name);                           \
  if (!call_##name)                                                            \
    return 3;
int main(int argc, char **argv) {
  if (argc != 4)
    return 2;
  if (cuInit(0) != CUDA_SUCCESS)
    return 10;
  CUdevice device;
  unsigned initial_flags = 0;
  int active = 0;
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxGetState(device, &initial_flags, &active) !=
          CUDA_SUCCESS ||
      active ||
      cuDevicePrimaryCtxSetFlags(device, (initial_flags & ~7u) |
                                             CU_CTX_SCHED_BLOCKING_SYNC) !=
          CUDA_SUCCESS)
    return 10;

  if (!dlopen("/app/libggml.so", RTLD_NOW | RTLD_GLOBAL) ||
      !dlopen("/app/libggml-cuda.so", RTLD_NOW | RTLD_GLOBAL))
    return 3;
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
  unsigned char weights[64 * 1440], activation[2880];
  float input[2560], output[64];
  FILE *f = fopen(argv[1], "rb");
  if (!f)
    return 4;
  if (fread(weights, 1, sizeof(weights), f) != sizeof(weights))
    return 4;
  fclose(f);
  f = fopen(argv[2], "rb");
  if (!f)
    return 4;
  if (fread(activation, 1, sizeof(activation), f) != sizeof(activation))
    return 4;
  fclose(f);
  for (int group = 0; group < 80; group++) {
    uint16_t delta;
    memcpy(&delta, activation + group * 36, 2);
    if (delta != 0x2000)
      return 4;
    for (int i = 0; i < 32; i++)
      input[group * 32 + i] =
          (float)(int8_t)activation[group * 36 + 4 + i] / 128.0f;
    if (input[group * 32] != 127.0f / 128.0f)
      return 4;
  }
  struct ggml_init_params params = {
      .mem_size = 8 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = true};
  struct ggml_context *ctx = call_ggml_init(params);
  if (!ctx)
    return 5;
  ggml_backend_t backend = call_ggml_backend_cuda_init(0);
  if (!backend)
    return 6;
  struct ggml_tensor *a =
      call_ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2560, 64);
  struct ggml_tensor *b = call_ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
  struct ggml_tensor *result = call_ggml_mul_mat(ctx, a, b);
  struct ggml_cgraph *graph = call_ggml_new_graph_custom(ctx, 64, false);
  call_ggml_build_forward_expand(graph, result);
  ggml_backend_buffer_t buffer =
      call_ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (!buffer)
    return 7;
  void *driver = dlopen("libcuda.so.1", RTLD_NOW);
  unsigned flags = 0;
  int (*get_flags)(unsigned *) =
      driver ? (int (*)(unsigned *))dlsym(driver, "cuCtxGetFlags") : NULL;
  if (!get_flags || get_flags(&flags) || (flags & 7) != 4)
    return 10;
  call_ggml_backend_tensor_set(a, weights, 0, sizeof(weights));
  call_ggml_backend_tensor_set(b, input, 0, sizeof(input));
  int status = call_ggml_backend_graph_compute(backend, graph);
  if (!status) {
    call_ggml_backend_tensor_get(result, output, 0, sizeof(output));
    f = fopen(argv[3], "wb");
    if (!f)
      status = 8;
    else {
      if (fwrite(output, sizeof(float), 64, f) != 64)
        status = 9;
      fclose(f);
    }
  }
  call_ggml_backend_buffer_free(buffer);
  call_ggml_backend_free(backend);
  call_ggml_free(ctx);
  return status;
}
