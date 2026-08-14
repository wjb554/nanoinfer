/// Continuous Batching Demo — 3 concurrent users through EngineServer +
/// BatchMainLoop (real continuous batching: prefill + decode in one batch).
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include "lightllm/engine/engine.h"
#include "lightllm/engine/batch_loop.h"

using namespace lightllm::engine;
using namespace lightllm::kv_cache;

int main(int argc, char** argv) {
    const char* model_dir = argc > 1 ? argv[1] : "models/qwen2.5-0.5b";
    bool use_fp16 = false;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--fp16")) use_fp16 = true;

    printf("=== Continuous Batching (3 users) ===\n");

    EngineServer engine(model_dir, /*max_seq_len=*/0, /*max_batch_tokens=*/256,
                        /*kv_cache_mb=*/0, prefix_cache_policy_from_env(), use_fp16);
    BatchMainLoop loop(engine, SchedulerPolicy::DecodeFirst, /*chunk_size=*/16,
                       /*max_batch_tokens=*/256);

    // 3 users with different prompts (same as the original demo).
    std::vector<std::vector<int>> prompts = {
        {576, 8319, 315, 13466, 374},   // "The capital of France is"
        {15837, 467, 315, 1605},        // "The weather today is"
        {10585, 374, 279, 18865},       // "I think therefore I"
    };
    std::vector<int> ids;
    for (auto& p : prompts)
        ids.push_back(loop.submit(p, /*max_new=*/16, /*eos=*/151643, "", "",
                                  /*temperature=*/0.8f));

    auto t0 = std::chrono::steady_clock::now();
    loop.run();
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    printf("\n=== Results ===\n");
    int total = 0;
    for (size_t i = 0; i < ids.size(); i++) {
        auto gen = loop.generated_tokens(ids[i]);
        total += static_cast<int>(gen.size());
        printf("User %zu: %zu -> %zu tokens\n",
               i, prompts[i].size(), gen.size());
        printf("  tokens:");
        for (int t : gen) printf(" %d", t);
        printf("\n");
    }
    printf("\n%d tokens across %zu users in %.0f ms (%.1f tok/s batched)\n",
           total, ids.size(), ms, total * 1000.0 / (ms > 0 ? ms : 1.0));
    return 0;
}
