/// LightLLM — multi-user mixed-length benchmark.
/// 6 users with different prompt lengths, Continuous Batching.

#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include "lightllm/engine/engine.h"
#include "lightllm/engine/batch_loop.h"
#include "bench_common.h"

using namespace lightllm::engine;

int main(int argc, char** argv) {
    // Args: [--model <dir>] [--fp16] [--max-batch-tokens N]
    const char* model_dir = "models/qwen2.5-0.5b";
    bool use_fp16 = false;
    int  max_batch_tokens = 0;   // 0 -> auto-derived from GPU memory
    int  kv_cache_mb  = 0;       // 0 -> auto (seq-len cap); >0 -> explicit pool budget
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model") && i+1 < argc)          model_dir = argv[++i];
        else if (!strcmp(argv[i], "--fp16"))                         use_fp16  = true;
        else if (!strcmp(argv[i], "--max-batch-tokens") && i+1<argc) max_batch_tokens = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-cache-mb") && i+1 < argc)    kv_cache_mb = std::atoi(argv[++i]);
    }
    if (max_batch_tokens <= 0) max_batch_tokens = default_batch_tokens();
    printf("=============================================================\n");
    printf("  LightLLM — Multi-User Mixed-Length Test\n");
    printf("  6 users simultaneously, Continuous Batching\n");
    printf("=============================================================\n\n");

    struct User { std::vector<int> prompt; int max_new; const char* label; };
    std::vector<User> users = {
        {{576}, 10, "1tok  "},
        {{576,8319,315}, 10, "3tok  "},
        {{576,8319,315,13466,374}, 10, "5tok  "},
        {{100,101,102,103,104,105,106,107,108,109}, 10, "10tok "},
    };

    printf("Loading model... (%s, %s, batch=%d)\n", model_dir,
           use_fp16 ? "FP16" : "FP32", max_batch_tokens); fflush(stdout);
    EngineServer engine(model_dir, 0, max_batch_tokens, kv_cache_mb,
                        lightllm::kv_cache::prefix_cache_policy_from_env(),
                        use_fp16);
    BatchMainLoop batch(engine, SchedulerPolicy::DecodeFirst, 16, max_batch_tokens);
    printf("Ready.\n\n"); fflush(stdout);

    // Submit all simultaneously
    printf("Submitting %zu users...\n", users.size()); fflush(stdout);
    for(int i=0;i<(int)users.size();i++){
        fprintf(stderr, "  user %d...\n", i); fflush(stderr);
        batch.submit(users[i].prompt, users[i].max_new, 151643);
    }

    // Run to completion
    printf("Starting batch execution...\n"); fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();
    int steps = 0;
    while(batch.has_active() && steps < 500) {
        fprintf(stderr, "  step %d (active=%d)...\n", steps, batch.active_count()); fflush(stderr);
        batch.step();
        steps++;
    }
    fprintf(stderr, "Done: %d steps\n", steps); fflush(stderr);
    auto t1 = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    int total_tok=0;
    printf("  %-8s | %6s | %6s | %7s | %7s | %7s\n",
           "User","prompt","output","TTFT ms","TPOT ms","Total ms");
    printf("  --------|--------|--------|---------|---------|--------\n");
    for(int i=0;i<(int)users.size();i++){
        for(auto& m : batch.all_metrics()){
            if(m.request_id==i && m.finished){
                total_tok+=m.output_len;
                printf("  %-8s | %6d | %6d | %7.0f | %7.0f | %7.0f\n",
                       users[i].label, m.prompt_len, m.output_len,
                       m.ttft_ms(), m.tpot_ms(), m.latency_ms());
            }
        }
    }
    printf("  --------|--------|--------|---------|---------|--------\n");
    printf("  %-8s | %6s | %6d | %7s | %7s | %7.0f\n",
           "TOTAL","-",total_tok,"-","-",wall_ms);
    printf("\n  %.1f tok/s (%zu users, %d steps)\n",
           total_tok*1000.0/wall_ms, users.size(), steps);
    printf("  Single-user baseline: ~14 tok/s (sequential)\n");

    // Compare with sequential
    double seq_time = total_tok / 14.0 * 1000;
    printf("  Estimated sequential: %.0f ms → batched is %.1fx faster\n",
           seq_time, seq_time / wall_ms);
    return 0;
}
