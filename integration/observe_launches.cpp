#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

struct Fusion {
    const void *x_bias, *gate, *gate_bias, *x_scale, *gate_scale;
    int glu_op;
    float glu_limit;
    const void *shared_up, *shared_gate;
    float *shared_dst;
    uint32_t shared_stride_col_dst;
};
static_assert(sizeof(Fusion) == 80);
static std::atomic<unsigned long> matches{0};
static const char *target = "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";

template<class T> static T symbol(const char *name) {
    void *address = dlsym(RTLD_NEXT, name);
    if (!address) {
        static void *runtime = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
        if (runtime) address = dlsym(runtime, name);
    }
    auto value = reinterpret_cast<T>(address);
    if (!value) { std::fprintf(stderr, "mojo_observer missing=%s\n", name); std::abort(); }
    return value;
}
template<class T> static T argument(void **args, int i) {
    T value;
    std::memcpy(&value, args[i], sizeof(value));
    return value;
}
static void observe(const void *func, dim3 grid, dim3 block, void **args, const char *api) {
    if (grid.x != 9728 || grid.y != 1 || grid.z != 1 || block.x != 32 || block.y != 4 || block.z != 1) return;
    static auto get_name = symbol<decltype(&cudaFuncGetName)>("cudaFuncGetName");
    const char *name = nullptr;
    if (get_name(&name, func) != cudaSuccess || !name || std::strcmp(name, target)) return;
    auto count = ++matches;
    if (count > 4) return;
    const auto fusion = argument<Fusion>(args, 3);
    std::fprintf(stderr,
        "mojo_observer api=%s match=%lu width=%u rows=%u weight_stride=%u activation_stride=%u "
        "ids=%d gate=%d x_bias=%d gate_bias=%d x_scale=%d gate_scale=%d shared_up=%d shared_gate=%d shared_dst=%d glu=%d\n",
        api, count, argument<uint32_t>(args,5), argument<uint32_t>(args,9),
        argument<uint32_t>(args,7), argument<uint32_t>(args,8),
        argument<const void *>(args,2)!=nullptr, fusion.gate!=nullptr,
        fusion.x_bias!=nullptr, fusion.gate_bias!=nullptr, fusion.x_scale!=nullptr,
        fusion.gate_scale!=nullptr, fusion.shared_up!=nullptr, fusion.shared_gate!=nullptr,
        fusion.shared_dst!=nullptr, fusion.glu_op);
}
extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 grid, dim3 block,
                                       void **args, size_t bytes, cudaStream_t stream) {
    static auto real = symbol<decltype(&cudaLaunchKernel)>("cudaLaunchKernel");
    observe(func,grid,block,args,"launch");
    return real(func,grid,block,args,bytes,stream);
}
extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args) {
    static auto real = symbol<decltype(&cudaLaunchKernelExC)>("cudaLaunchKernelExC");
    observe(func,config->gridDim,config->blockDim,args,"launch_ex");
    return real(config,func,args);
}
__attribute__((destructor)) static void report() {
    std::fprintf(stderr,"mojo_observer target_launch_calls=%lu mode=observe_only\n",matches.load());
}
