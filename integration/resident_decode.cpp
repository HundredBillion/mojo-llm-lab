#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static std::map<std::string,long long> cpu_stat() {
    std::ifstream f("/sys/fs/cgroup/cpu.stat");
    if (!f) throw std::runtime_error("cpu.stat unavailable");
    std::map<std::string,long long> result;
    std::string key; long long value;
    while (f >> key >> value) result[key]=value;
    for (const auto *key : {"usage_usec","nr_throttled","throttled_usec"})
        if (!result.count(key)) throw std::runtime_error("CPU counter missing");
    return result;
}
static long long ns(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
}
int main(int argc,char **argv) try {
    if (argc!=3) throw std::runtime_error("model and control|mojo required");
    const std::string mode=argv[2];
    if (mode!="control" && mode!="mojo") throw std::runtime_error("invalid mode");
    const char *path=std::getenv("MOJO_CANDIDATE_PTX");
    if (mode=="mojo" && !path) throw std::runtime_error("candidate required");
    auto count=reinterpret_cast<unsigned long(*)()>(dlsym(RTLD_DEFAULT,"mojo_replacement_count"));
    auto captured=reinterpret_cast<unsigned long(*)()>(dlsym(RTLD_DEFAULT,"mojo_capture_count"));
    if (!count || !captured) throw std::runtime_error("observer counters required");
    auto select=[&](int arm) {
        if (arm==1 && mode=="mojo") setenv("MOJO_REPLACE_PTX",path,1);
        else unsetenv("MOJO_REPLACE_PTX");
    };
    unsetenv("MOJO_REPLACE_PTX");
    ggml_backend_load_all_from_path("/app");
    llama_backend_init();
    auto cpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) throw std::runtime_error("CPU backend missing");
    auto reg=ggml_backend_dev_backend_reg(cpu);
    auto make_pool=reinterpret_cast<decltype(&ggml_threadpool_new)>(ggml_backend_reg_get_proc_address(reg,"ggml_threadpool_new"));
    auto free_pool=reinterpret_cast<decltype(&ggml_threadpool_free)>(ggml_backend_reg_get_proc_address(reg,"ggml_threadpool_free"));
    if (!make_pool || !free_pool) throw std::runtime_error("threadpool API unavailable");
    auto tp=ggml_threadpool_params_default(2); tp.poll=0;
    auto pool=make_pool(&tp);
    if (!pool) throw std::runtime_error("threadpool creation failed");
    auto mp=llama_model_default_params(); mp.n_gpu_layers=99;
    auto *model=llama_model_load_from_file(argv[1],mp);
    if (!model) throw std::runtime_error("model load failed");
    auto cp=llama_context_default_params();
    cp.n_ctx=512; cp.n_batch=32; cp.n_ubatch=32;
    cp.n_threads=2; cp.n_threads_batch=2;
    cp.type_k=GGML_TYPE_Q4_0; cp.type_v=GGML_TYPE_Q4_0;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context *contexts[2]={llama_init_from_model(model,cp),llama_init_from_model(model,cp)};
    if (!contexts[0] || !contexts[1]) throw std::runtime_error("context creation failed");
    for (auto *ctx:contexts) llama_attach_threadpool(ctx,pool,pool);
    const auto *vocab=llama_model_get_vocab(model);
    const std::string text="Invoice for Oak Street Bakery. The statement lists coffee, rent, supplies and tax. The total amount includes all transactions. Review the business name and total carefully. Invoice for Oak Street Bakery. The statement lists coffee, rent, supplies and tax.";
    std::vector<llama_token> tokens(256);
    int n=llama_tokenize(vocab,text.data(),text.size(),tokens.data(),tokens.size(),true,true);
    if (n<40) throw std::runtime_error("fixture tokenization failed");
    std::cout << "{\"ready\":true,\"fixture_tokens\":" << n << "}" << std::endl;
    int arm,steps; int trial=0;
    while (std::cin >> arm >> steps) {
        if (arm<0 || arm>1 || steps<1 || steps>32) throw std::runtime_error("invalid trial");
        auto *ctx=contexts[arm]; select(arm);
        llama_memory_clear(llama_get_memory(ctx),true);
        if (llama_decode(ctx,llama_batch_get_one(tokens.data(),8))) throw std::runtime_error("prefill failed");
        llama_synchronize(ctx);
        const auto before=cpu_stat(); const auto begin=Clock::now();
        for (int i=0;i<steps;++i)
            if (llama_decode(ctx,llama_batch_get_one(tokens.data()+8+i,1))) throw std::runtime_error("decode failed");
        llama_synchronize(ctx);
        const auto elapsed=ns(begin); const auto after=cpu_stat();
        const auto *logits=llama_get_logits_ith(ctx,-1);
        const int nv=llama_vocab_n_tokens(vocab); int top=0;
        for (int i=0;i<nv;++i) {
            if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
            if (logits[i]>logits[top]) top=i;
        }
        std::cout << "{\"trial\":" << trial++ << ",\"arm\":" << arm
                  << ",\"tokens\":" << steps << ",\"decode_ns\":" << elapsed
                  << ",\"top_token\":" << top << ",\"replacement_calls\":" << count()
                  << ",\"captured_calls\":" << captured()
                  << ",\"cpu_usage_usec\":" << after.at("usage_usec")-before.at("usage_usec")
                  << ",\"nr_throttled\":" << after.at("nr_throttled")-before.at("nr_throttled")
                  << ",\"throttled_usec\":" << after.at("throttled_usec")-before.at("throttled_usec") << "}" << std::endl;
    }
    for (auto *ctx:contexts) { llama_detach_threadpool(ctx); llama_free(ctx); }
    free_pool(pool);
    llama_model_free(model); llama_backend_free();
    return 0;
} catch (const std::exception &e) { std::fprintf(stderr,"resident_error: %s\n",e.what()); return 1; }
