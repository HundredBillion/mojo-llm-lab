#include "llama.h"
#include "ggml-backend.h"
#ifdef MOJO_VALIDATE_LOGITS
#include "logit_validation.h"
#endif
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
    if (argc<3 || argc>5) throw std::runtime_error("model, control|mojo and optional priming and candidate-context arguments required");
    const std::string prime_arg=argc>=4 ? argv[3] : "0";
    if (prime_arg!="0" && prime_arg!="1" && prime_arg!="2") throw std::runtime_error("invalid priming flag");
    const int prime_tokens=std::stoi(prime_arg);
    const std::string arm_arg=argc==5 ? argv[4] : "1";
    if (arm_arg!="0" && arm_arg!="1") throw std::runtime_error("invalid candidate context");
    const int mojo_arm=std::stoi(arm_arg);
    const std::string mode=argv[2];
    if (mode!="control" && mode!="mojo") throw std::runtime_error("invalid mode");
    const char *path=std::getenv("MOJO_CANDIDATE_PTX");
    if (mode=="mojo" && !path) throw std::runtime_error("candidate required");
    auto count=reinterpret_cast<unsigned long(*)()>(dlsym(RTLD_DEFAULT,"mojo_replacement_count"));
    auto captured=reinterpret_cast<unsigned long(*)()>(dlsym(RTLD_DEFAULT,"mojo_capture_count"));
    if (!count || !captured) throw std::runtime_error("observer counters required");
    auto select=[&](int arm) {
        if (arm==mojo_arm && mode=="mojo") setenv("MOJO_REPLACE_PTX",path,1);
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
    const std::string original_text="Invoice for Oak Street Bakery. The statement lists coffee, rent, supplies and tax. The total amount includes all transactions. Review the business name and total carefully. Invoice for Oak Street Bakery. The statement lists coffee, rent, supplies and tax.";
    std::vector<llama_token> original_tokens(256);
    const int original_n=llama_tokenize(vocab,original_text.data(),original_text.size(),original_tokens.data(),original_tokens.size(),true,true);
    if (original_n<42) throw std::runtime_error("original fixture tokenization failed");
    std::string text=original_text;
    for (int repeat=1;repeat<8;++repeat) text+=" "+original_text;
    std::vector<llama_token> tokens(1024);
    int n=llama_tokenize(vocab,text.data(),text.size(),tokens.data(),tokens.size(),true,true);
    if (n<138 || !std::equal(original_tokens.begin(),original_tokens.begin()+42,tokens.begin()))
        throw std::runtime_error("extended fixture capacity or prefix mismatch");
    std::cout << "{\"ready\":true,\"fixture_tokens\":" << n << ",\"max_timed_tokens\":128,\"prime_tokens\":" << prime_tokens << ",\"mojo_arm\":" << mojo_arm << "}" << std::endl;
#ifdef MOJO_VALIDATE_LOGITS
    if (mode!="mojo") throw std::runtime_error("logit validation requires Mojo comparison");
    const int validation_vocab=llama_vocab_n_tokens(vocab);
    if (validation_vocab<=0) throw std::runtime_error("invalid validation vocabulary");
    LogitValidation validation(validation_vocab);
    int validation_steps=-1;
    size_t candidate_trials=0,cuda_trials=0;
    auto observe_logits=[&](llama_context *ctx,bool candidate,size_t position) {
        llama_synchronize(ctx);
        validation.observe(candidate,position,llama_get_logits_ith(ctx,-1),validation_vocab);
    };
#endif
    int arm,steps; int trial=0;
    while (std::cin >> arm >> steps) {
        if (arm<0 || arm>1 || steps<1 || steps>128 || 8+prime_tokens+steps>n) throw std::runtime_error("invalid trial");
#ifdef MOJO_VALIDATE_LOGITS
        if (validation_steps>=0 && validation_steps!=steps) throw std::runtime_error("validation workload changed");
        validation_steps=steps;
        const bool candidate=arm==mojo_arm;
#endif
        auto *ctx=contexts[arm]; select(arm);
        llama_memory_clear(llama_get_memory(ctx),true);
        if (llama_decode(ctx,llama_batch_get_one(tokens.data(),8))) throw std::runtime_error("prefill failed");
        llama_synchronize(ctx);
#ifdef MOJO_VALIDATE_LOGITS
        observe_logits(ctx,candidate,0);
#endif
        for (int prime=0;prime<prime_tokens;++prime) {
            if (llama_decode(ctx,llama_batch_get_one(tokens.data()+8+prime,1))) throw std::runtime_error("priming decode failed");
            llama_synchronize(ctx);
#ifdef MOJO_VALIDATE_LOGITS
            observe_logits(ctx,candidate,1+prime);
#endif
        }
        const auto calls_before=count(), captures_before=captured();
        const auto before=cpu_stat(); const auto begin=Clock::now();
        for (int i=0;i<steps;++i) {
            if (llama_decode(ctx,llama_batch_get_one(tokens.data()+8+prime_tokens+i,1))) throw std::runtime_error("decode failed");
#ifdef MOJO_VALIDATE_LOGITS
            observe_logits(ctx,candidate,1+prime_tokens+i);
#endif
        }
        llama_synchronize(ctx);
#ifdef MOJO_VALIDATE_LOGITS
        if (candidate) ++candidate_trials; else ++cuda_trials;
#endif
        const auto elapsed=ns(begin); const auto after=cpu_stat();
        const auto timed_calls=count()-calls_before, timed_captures=captured()-captures_before;
        const auto *logits=llama_get_logits_ith(ctx,-1);
        const int nv=llama_vocab_n_tokens(vocab); int top=0;
        for (int i=0;i<nv;++i) {
            if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
            if (logits[i]>logits[top]) top=i;
        }
        std::cout << "{\"trial\":" << trial++ << ",\"arm\":" << arm
                  << ",\"mojo_arm\":" << mojo_arm
                  << ",\"prime_tokens\":" << prime_tokens
                  << ",\"tokens\":" << steps << ",\"decode_ns\":" << elapsed
                  << ",\"top_token\":" << top << ",\"replacement_calls\":" << count()
                  << ",\"captured_calls\":" << captured()
                  << ",\"timed_replacement_calls\":" << timed_calls
                  << ",\"timed_captured_calls\":" << timed_captures
                  << ",\"cpu_usage_usec\":" << after.at("usage_usec")-before.at("usage_usec")
                  << ",\"nr_throttled\":" << after.at("nr_throttled")-before.at("nr_throttled")
                  << ",\"throttled_usec\":" << after.at("throttled_usec")-before.at("throttled_usec") << "}" << std::endl;
    }
#ifdef MOJO_VALIDATE_LOGITS
    const size_t positions=1+prime_tokens+validation_steps;
    const bool valid=validation.passed(positions,candidate_trials,cuda_trials);
    validation.report(stderr,valid,positions);
    if (!valid) throw std::runtime_error("whole-logit validation failed");
#endif
    for (auto *ctx:contexts) { llama_detach_threadpool(ctx); llama_free(ctx); }
    free_pool(pool);
    llama_model_free(model); llama_backend_free();
    return 0;
} catch (const std::exception &e) { std::fprintf(stderr,"resident_error: %s\n",e.what()); return 1; }
