#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <mutex>
#include <cstddef>
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
static_assert(offsetof(Fusion,gate)==8 && offsetof(Fusion,glu_op)==40 && offsetof(Fusion,shared_dst)==64);
static std::atomic<unsigned long> matches{0}, eligible_calls{0}, validated{0};
static std::mutex validation_mutex;
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
static bool eligible(void **args) {
    const auto f = argument<Fusion>(args, 3);
    return argument<uint32_t>(args,5)==2560 && argument<uint32_t>(args,9)==9728 &&
        argument<uint32_t>(args,7)==10 && argument<uint32_t>(args,8)==80 &&
        !argument<const void *>(args,2) && argument<const void *>(args,0) &&
        argument<const void *>(args,1) && argument<void *>(args,4) && f.gate &&
        !f.x_bias && !f.gate_bias && !f.x_scale && !f.gate_scale &&
        !f.shared_up && !f.shared_gate && !f.shared_dst && f.glu_op==2 && f.glu_limit==0;
}
template<class T> static T driver(const char *name) {
    static void *library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    auto fn = reinterpret_cast<T>(library ? dlsym(library,name) : nullptr);
    if (!fn) { std::fprintf(stderr,"mojo_validation missing=%s\n",name); std::abort(); }
    return fn;
}
static void checked(int code, const char *operation) {
    if (code) { std::fprintf(stderr,"mojo_validation error=%d operation=%s\n",code,operation); std::abort(); }
}
static void validate(void **args, cudaStream_t stream) {
    const char *path = std::getenv("MOJO_VALIDATE_PTX");
    if (!path) return;
    std::lock_guard<std::mutex> lock(validation_mutex);
    static const unsigned long limit = [] {
        const char *value=std::getenv("MOJO_VALIDATE_CALLS");
        if (!value) return 36ul;
        char *end=nullptr;
        unsigned long count=std::strtoul(value,&end,10);
        if (!*value || *end || count<1 || count>10000) std::abort();
        return count;
    }();
    if (validated.load() >= limit) return;
    cudaStreamCaptureStatus capture;
    checked(symbol<decltype(&cudaStreamIsCapturing)>("cudaStreamIsCapturing")(stream,&capture),"capture_status");
    if (capture != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr,"mojo_validation requires_graphs_disabled\n"); std::abort();
    }
    checked(symbol<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize")(stream),"original_sync");
    CUmodule module;
    CUfunction kernel;
    const char *entry = std::getenv("MOJO_VALIDATE_ENTRY");
    if (!entry) std::abort();
    checked(driver<decltype(&cuModuleLoad)>("cuModuleLoad")(&module,path),"module_load");
    checked(driver<decltype(&cuModuleGetFunction)>("cuModuleGetFunction")(&kernel,module,entry),"function");
    constexpr size_t rows=9728, bytes=(rows+2)*sizeof(float);
    void *allocation=nullptr;
    checked(symbol<decltype(&cudaMalloc)>("cudaMalloc")(&allocation,bytes),"allocate");
    checked(symbol<decltype(&cudaMemset)>("cudaMemset")(allocation,0xff,bytes),"sentinels");
    auto output=static_cast<float *>(allocation)+1;
    auto up=argument<const void *>(args,0), input=argument<const void *>(args,1);
    auto gate=argument<Fusion>(args,3).gate;
    void *parameters[]={&up,&gate,&input,&output};
    checked(driver<decltype(&cuLaunchKernel)>("cuLaunchKernel")(kernel,rows,1,1,128,1,1,0,
        reinterpret_cast<CUstream>(stream),parameters,nullptr),"launch");
    checked(symbol<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize")(stream),"candidate_sync");
    std::vector<float> reference(rows), candidate(rows+2);
    auto copy=symbol<decltype(&cudaMemcpy)>("cudaMemcpy");
    checked(copy(reference.data(),argument<void *>(args,4),rows*sizeof(float),cudaMemcpyDeviceToHost),"reference_copy");
    checked(copy(candidate.data(),allocation,bytes,cudaMemcpyDeviceToHost),"candidate_copy");
    uint32_t before,after;
    std::memcpy(&before,candidate.data(),4); std::memcpy(&after,candidate.data()+rows+1,4);
    if (before!=0xffffffffu || after!=0xffffffffu) { std::fprintf(stderr,"mojo_validation canary_failed\n"); std::abort(); }
    float maximum=0;
    for (size_t i=0;i<rows;++i) {
        float ref=reference[i], actual=candidate[i+1], difference=std::fabs(ref-actual);
        if (!std::isfinite(ref) || !std::isfinite(actual) || difference>1e-5f+1e-5f*std::fabs(ref)) {
            std::fprintf(stderr,"mojo_validation mismatch call=%lu row=%zu reference=%.9g candidate=%.9g\n",
                validated.load()+1,i,ref,actual); std::abort();
        }
        maximum=std::fmax(maximum,difference);
    }
    checked(symbol<decltype(&cudaFree)>("cudaFree")(allocation),"free");
    checked(driver<decltype(&cuModuleUnload)>("cuModuleUnload")(module),"module_unload");
    std::fprintf(stderr,"mojo_validation passed_call=%lu rows=%zu max_abs=%.9g\n",++validated,rows,maximum);
}
static bool observe(const void *func, dim3 grid, dim3 block, void **args, const char *api) {
    if (grid.x != 9728 || grid.y != 1 || grid.z != 1 || block.x != 32 || block.y != 4 || block.z != 1) return false;
    static auto get_name = symbol<decltype(&cudaFuncGetName)>("cudaFuncGetName");
    const char *name = nullptr;
    if (get_name(&name, func) != cudaSuccess || !name || std::strcmp(name, target)) return false;
    auto count = ++matches;
    const bool supported = eligible(args);
    if (supported) ++eligible_calls;
    if (count > 4) return supported;
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
    return supported;
}
extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 grid, dim3 block,
                                       void **args, size_t bytes, cudaStream_t stream) {
    static auto real = symbol<decltype(&cudaLaunchKernel)>("cudaLaunchKernel");
    bool supported=observe(func,grid,block,args,"launch");
    auto status=real(func,grid,block,args,bytes,stream);
    if (status==cudaSuccess && supported) validate(args,stream);
    return status;
}
extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args) {
    static auto real = symbol<decltype(&cudaLaunchKernelExC)>("cudaLaunchKernelExC");
    bool supported=observe(func,config->gridDim,config->blockDim,args,"launch_ex");
    auto status=real(config,func,args);
    if (status==cudaSuccess && supported) validate(args,config->stream);
    return status;
}
__attribute__((destructor)) static void report() {
    std::fprintf(stderr,"mojo_validation eligible_calls=%lu validated_calls=%lu outputs=CUDA\n",eligible_calls.load(),validated.load());
    std::fprintf(stderr,"mojo_observer target_launch_calls=%lu mode=observe_only\n",matches.load());
}
