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

#include "fusion_eligibility.h"
static const Layout &layout() {
    const char *name=std::getenv("MOJO_SHAPE");
    if (!name || std::strcmp(name,"qwen3-4b")==0) return layout_4b;
    if (std::strcmp(name,"qwen35-2b")==0) return layout_2b;
    std::fprintf(stderr,"mojo_observer unsupported_shape\n"); std::abort();
}
static std::atomic<unsigned long> matches{0}, eligible_calls{0}, validated{0};
static std::mutex validation_mutex, replacement_mutex;
static std::atomic<unsigned long> replaced{0}, captured_replacements{0};
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
template<class T> static T driver(const char *name) {
    static void *library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    auto fn = reinterpret_cast<T>(library ? dlsym(library,name) : nullptr);
    if (!fn) { std::fprintf(stderr,"mojo_validation missing=%s\n",name); std::abort(); }
    return fn;
}
static void checked(int code, const char *operation) {
    if (code) { std::fprintf(stderr,"mojo_validation error=%d operation=%s\n",code,operation); std::abort(); }
}
static CUcontext replacement_context=nullptr;
static CUmodule replacement_module=nullptr;
static CUfunction replacement_kernel=nullptr;
static bool prepare_replacement() {
    const char *path=std::getenv("MOJO_REPLACE_PTX");
    if (!path) return false;
    if (std::getenv("MOJO_VALIDATE_PTX")) std::abort();
    static auto current=driver<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent");
    CUcontext context=nullptr;
    checked(current(&context),"current_context");
    if (!context) return false;
    std::lock_guard<std::mutex> lock(replacement_mutex);
    if (replacement_kernel) return context==replacement_context;
    CUdevice device;
    int major,minor;
    checked(driver<decltype(&cuCtxGetDevice)>("cuCtxGetDevice")(&device),"device");
    auto attribute=driver<decltype(&cuDeviceGetAttribute)>("cuDeviceGetAttribute");
    checked(attribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device),"major");
    checked(attribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device),"minor");
    if (major!=8 || minor!=6) return false;
    const char *entry=std::getenv("MOJO_REPLACE_ENTRY");
    if (!entry) std::abort();
    checked(driver<decltype(&cuModuleLoad)>("cuModuleLoad")(&replacement_module,path),"replacement_module");
    checked(driver<decltype(&cuModuleGetFunction)>("cuModuleGetFunction")(&replacement_kernel,replacement_module,entry),"replacement_function");
    int registers;
    checked(driver<decltype(&cuFuncGetAttribute)>("cuFuncGetAttribute")(&registers,CU_FUNC_ATTRIBUTE_NUM_REGS,replacement_kernel),"replacement_ready");
    replacement_context=context;
    return true;
}
static bool replace(void **args,cudaStream_t stream) {
    if (!std::getenv("MOJO_REPLACE_PTX")) return false;
    static auto capturing=symbol<decltype(&cudaStreamIsCapturing)>("cudaStreamIsCapturing");
    cudaStreamCaptureStatus capture;
    checked(capturing(stream,&capture),"replacement_capture_status");
    if (capture==cudaStreamCaptureStatusNone) {
        if (!prepare_replacement()) return false;
    }
    std::lock_guard<std::mutex> lock(replacement_mutex);
    CUcontext current=nullptr;
    static auto context=driver<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent");
    checked(context(&current),"replacement_context");
    if (!replacement_kernel || current!=replacement_context) return false;
    auto up=argument<const void *>(args,0),input=argument<const void *>(args,1);
    auto gate=argument<Fusion>(args,3).gate;
    auto output=argument<void *>(args,4);
    void *parameters[]={&up,&gate,&input,&output};
    static auto launch=driver<decltype(&cuLaunchKernel)>("cuLaunchKernel");
    checked(launch(replacement_kernel,layout().rows,1,1,128,1,1,0,reinterpret_cast<CUstream>(stream),parameters,nullptr),"replacement_launch");
    ++replaced;
    if (capture==cudaStreamCaptureStatusActive) ++captured_replacements;
    return true;
}
extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream,cudaStreamCaptureMode mode) {
    static auto real=symbol<decltype(&cudaStreamBeginCapture)>("cudaStreamBeginCapture");
    prepare_replacement();
    return real(stream,mode);
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
    const size_t rows=layout().rows, bytes=(rows+2)*sizeof(float);
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
    if (std::getenv("MOJO_VALIDATE_FEEDBACK")) {
        checked(copy(argument<void *>(args,4),output,rows*sizeof(float),cudaMemcpyDeviceToDevice),"feedback_copy");
    }
    checked(symbol<decltype(&cudaFree)>("cudaFree")(allocation),"free");
    checked(driver<decltype(&cuModuleUnload)>("cuModuleUnload")(module),"module_unload");
    std::fprintf(stderr,"mojo_validation passed_call=%lu rows=%zu max_abs=%.9g\n",++validated,rows,maximum);
}
static bool observe(const void *func, dim3 grid, dim3 block, void **args, const char *api) {
    if (grid.x != layout().rows || grid.y != 1 || grid.z != 1 || block.x != 32 || block.y != 4 || block.z != 1) return false;
    static auto get_name = symbol<decltype(&cudaFuncGetName)>("cudaFuncGetName");
    const char *name = nullptr;
    if (get_name(&name, func) != cudaSuccess || !name || std::strcmp(name, target)) return false;
    auto count = ++matches;
    const bool supported = eligible(args,layout());
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
    if (supported && bytes==0 && replace(args,stream)) return cudaSuccess;
    auto status=real(func,grid,block,args,bytes,stream);
    if (status==cudaSuccess && supported) validate(args,stream);
    return status;
}
extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args) {
    static auto real = symbol<decltype(&cudaLaunchKernelExC)>("cudaLaunchKernelExC");
    bool supported=observe(func,config->gridDim,config->blockDim,args,"launch_ex");
    if (supported && config->dynamicSmemBytes==0 && config->numAttrs==0 && replace(args,config->stream)) return cudaSuccess;
    auto status=real(config,func,args);
    if (status==cudaSuccess && supported) validate(args,config->stream);
    return status;
}
__attribute__((destructor)) static void report() {
    std::fprintf(stderr,"mojo_replacement calls=%lu captured_calls=%lu\n",replaced.load(),captured_replacements.load());
    std::fprintf(stderr,"mojo_validation eligible_calls=%lu validated_calls=%lu outputs=%s\n",eligible_calls.load(),validated.load(),(std::getenv("MOJO_VALIDATE_FEEDBACK") || std::getenv("MOJO_REPLACE_PTX")) ? "Mojo" : "CUDA");
    std::fprintf(stderr,"mojo_observer target_launch_calls=%lu mode=host_launches\n",matches.load());
}
