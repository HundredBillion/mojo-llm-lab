// Numerical-only coverage worker: full prompt prefill, then free-running greedy or seeded-sampled
// generation, comparing full logits between the Mojo candidate context and CUDA contexts.
// Timings from this worker are never performance evidence.
#include "llama.h"
#include "ggml-backend.h"
#include "logit_validation.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string json_escape(const std::string &text) {
    std::string out;
    for (unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) { char buffer[8]; std::snprintf(buffer, sizeof buffer, "\\u%04x", c); out += buffer; }
                else out += char(c);
        }
    }
    return out;
}

int main(int argc, char **argv) try {
    if (argc != 6) throw std::runtime_error("model, greedy|sampled, seed, steps and candidate-context arguments required");
    const std::string policy = argv[2];
    if (policy != "greedy" && policy != "sampled") throw std::runtime_error("invalid policy");
    const unsigned long seed = std::stoul(argv[3]);
    const int steps = std::stoi(argv[4]);
    if (steps < 1 || steps > 256) throw std::runtime_error("invalid steps");
    const std::string arm_arg = argv[5];
    if (arm_arg != "0" && arm_arg != "1") throw std::runtime_error("invalid candidate context");
    const int mojo_arm = std::stoi(arm_arg);
    const char *path = std::getenv("MOJO_CANDIDATE_PTX");
    if (!path) throw std::runtime_error("candidate required");
    auto count = reinterpret_cast<unsigned long (*)()>(dlsym(RTLD_DEFAULT, "mojo_replacement_count"));
    auto captured = reinterpret_cast<unsigned long (*)()>(dlsym(RTLD_DEFAULT, "mojo_capture_count"));
    if (!count || !captured) throw std::runtime_error("observer counters required");
    auto select = [&](int arm) {
        if (arm == mojo_arm) setenv("MOJO_REPLACE_PTX", path, 1);
        else unsetenv("MOJO_REPLACE_PTX");
    };
    unsetenv("MOJO_REPLACE_PTX");

    std::ifstream prompt_file("/prompt.txt", std::ios::binary);
    if (!prompt_file) throw std::runtime_error("prompt unavailable");
    std::stringstream prompt_stream; prompt_stream << prompt_file.rdbuf();
    const std::string prompt = prompt_stream.str();
    if (prompt.empty()) throw std::runtime_error("empty prompt");

    ggml_backend_load_all_from_path("/app");
    llama_backend_init();
    auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) throw std::runtime_error("CPU backend missing");
    auto reg = ggml_backend_dev_backend_reg(cpu);
    auto make_pool = reinterpret_cast<decltype(&ggml_threadpool_new)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
    auto free_pool = reinterpret_cast<decltype(&ggml_threadpool_free)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"));
    if (!make_pool || !free_pool) throw std::runtime_error("threadpool API unavailable");
    auto tp = ggml_threadpool_params_default(2); tp.poll = 0;
    auto pool = make_pool(&tp);
    if (!pool) throw std::runtime_error("threadpool creation failed");
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) throw std::runtime_error("model load failed");
    const auto *vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(prompt.size() + 16);
    const int n_prompt = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), false, true);
    if (n_prompt < 8) throw std::runtime_error("prompt tokenization failed");
    tokens.resize(n_prompt);
    const int n_ctx = ((n_prompt + steps + 16 + 255) / 256) * 256;
    if (n_ctx > 8192) throw std::runtime_error("prompt too long for 8192 context");

    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = 32; cp.n_ubatch = 32;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.type_k = GGML_TYPE_Q4_0; cp.type_v = GGML_TYPE_Q4_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context *contexts[2] = {llama_init_from_model(model, cp), llama_init_from_model(model, cp)};
    if (!contexts[0] || !contexts[1]) throw std::runtime_error("context creation failed");
    for (auto *ctx : contexts) llama_attach_threadpool(ctx, pool, pool);

    const int nv = llama_vocab_n_tokens(vocab);
    if (nv <= 0) throw std::runtime_error("invalid vocabulary");
    const size_t positions = size_t(steps) + 1;
    LogitValidation validation(nv, positions);
    size_t candidate_trials = 0, cuda_trials = 0;
    std::cout << "{\"ready\":true,\"prompt_tokens\":" << n_prompt << ",\"n_ctx\":" << n_ctx
              << ",\"steps\":" << steps << ",\"policy\":\"" << policy << "\",\"seed\":" << seed
              << ",\"mojo_arm\":" << mojo_arm << ",\"vocabulary\":" << nv << "}" << std::endl;

    int arm, trial = 0;
    while (std::cin >> arm) {
        if (arm < 0 || arm > 1) throw std::runtime_error("invalid trial");
        const bool candidate = arm == mojo_arm;
        auto *ctx = contexts[arm]; select(arm);
        const auto calls_before = count(), captures_before = captured();
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int offset = 0; offset < n_prompt; offset += 32) {
            const int length = std::min(32, n_prompt - offset);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + offset, length))) throw std::runtime_error("prefill failed");
        }
        llama_sampler *sampler = nullptr;
        if (policy == "sampled") {
            sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(uint32_t(seed)));
        }
        std::vector<llama_token> generated;
        int first_eog = -1;
        for (int i = 0; i <= steps; ++i) {
            llama_synchronize(ctx);
            const float *logits = llama_get_logits_ith(ctx, -1);
            validation.observe(candidate, size_t(i), logits, nv);
            if (i == steps) break;
            llama_token next;
            if (sampler) next = llama_sampler_sample(sampler, ctx, -1);
            else { next = 0; for (int v = 1; v < nv; ++v) if (logits[v] > logits[next]) next = v; }
            generated.push_back(next);
            if (first_eog < 0 && llama_vocab_is_eog(vocab, next)) first_eog = i;
            if (llama_decode(ctx, llama_batch_get_one(&generated.back(), 1))) throw std::runtime_error("generation decode failed");
        }
        if (sampler) llama_sampler_free(sampler);
        if (candidate) ++candidate_trials; else ++cuda_trials;
        std::string text;
        const int text_tokens = first_eog < 0 ? int(generated.size()) : first_eog;
        for (int i = 0; i < text_tokens; ++i) {
            char piece[256];
            const int n = llama_token_to_piece(vocab, generated[i], piece, sizeof piece, 0, false);
            if (n < 0) throw std::runtime_error("detokenization failed");
            text.append(piece, n);
        }
        std::cout << "{\"trial\":" << trial++ << ",\"arm\":" << arm << ",\"candidate\":" << (candidate ? "true" : "false")
                  << ",\"replacement_calls\":" << count() << ",\"captured_calls\":" << captured()
                  << ",\"trial_replacement_calls\":" << count() - calls_before
                  << ",\"trial_captured_calls\":" << captured() - captures_before
                  << ",\"first_eog\":" << first_eog << ",\"tokens\":[";
        for (size_t i = 0; i < generated.size(); ++i) std::cout << (i ? "," : "") << generated[i];
        std::cout << "],\"text\":\"" << json_escape(text) << "\"}" << std::endl;
    }
    const bool valid = validation.passed(positions, candidate_trials, cuda_trials);
    validation.report(stderr, valid, positions);
    if (!valid) throw std::runtime_error("whole-logit validation failed");
    for (auto *ctx : contexts) { llama_detach_threadpool(ctx); llama_free(ctx); }
    free_pool(pool);
    llama_model_free(model); llama_backend_free();
    return 0;
} catch (const std::exception &e) { std::fprintf(stderr, "resident_error: %s\n", e.what()); return 1; }
