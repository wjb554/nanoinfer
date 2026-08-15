/// Find max concurrent users. Submit N, run until idle. Binary search.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "nanoinfer/engine/engine.h"
#include "nanoinfer/engine/batch_loop.h"
#include "bench_common.h"
using namespace nanoinfer::engine;

int test_N(EngineServer& engine, int N, int max_batch_tokens) {
    BatchMainLoop batch(engine, SchedulerPolicy::DecodeFirst, 16, max_batch_tokens, 999);
    std::vector<int> prompt = {576, 8319, 315, 13466, 374};
    for (int i=0; i<N; i++) batch.submit(prompt, 4, 151643);

    int steps=0, max_steps=N*20;
    while (batch.has_active() && steps<max_steps) {
        batch.step(); steps++;
    }
    if (!batch.has_active()) {
        int tok=0;
        for (auto& m:batch.all_metrics()) if(m.finished) tok+=m.output_len;
        return tok;
    }
    return -1; // didn't finish = crash
}

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

    printf("=== Max Concurrent Users ===\n");
    printf("Model: %s | Precision: %s | MaxBatchTokens: %d | prompt=5 tok | gen=4 tok\n\n",
           model_dir, use_fp16 ? "FP16" : "FP32", max_batch_tokens);

    EngineServer engine(model_dir, 0, max_batch_tokens, kv_cache_mb,
                        nanoinfer::kv_cache::prefix_cache_policy_from_env(),
                        use_fp16);
    printf("KV blocks: %d\n\n", engine.num_blocks());
    fflush(stdout);

    int N_vals[] = {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
    int max_ok = 0;
    int n_vals_count = sizeof(N_vals)/sizeof(N_vals[0]);

    printf("  N     |  Status  |  Tokens |  Blocks\n");
    printf("  ------|----------|---------|--------\n");
    fflush(stdout);

    for (int ni=0; ni < n_vals_count; ni++) {
        int N = N_vals[ni];
        int tok = test_N(engine, N, max_batch_tokens);
        int free = engine.free_blocks();

        if (tok > 0) {
            printf("  %4d  |  OK      | %7d | %6d free\n", N, tok, free);
            max_ok = N;
        } else {
            printf("  %4d  |  FAIL    |       - |      -\n", N);
            break;
        }
        fflush(stdout);
    }

    printf("\n  Max concurrent stable: %d users\n", max_ok);
    printf("  Theoretical max (KV blocks): %d users (30tok/req)\n",
           engine.num_blocks() / 2);
    return 0;
}
