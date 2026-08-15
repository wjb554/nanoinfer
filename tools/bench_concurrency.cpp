/// NanoInfer — throughput vs concurrency (EngineServer + BatchMainLoop).
/// Measures TWO modes with the SAME engine so the batching benefit is visible:
///   Sequential — N requests run one at a time (no batching).
///   Batched    — N requests submitted together, continuous batching.
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include "nanoinfer/engine/engine.h"
#include "nanoinfer/engine/batch_loop.h"

using namespace nanoinfer::engine;
using namespace nanoinfer::kv_cache;

static double wall_ms(std::chrono::steady_clock::time_point t0) {
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    // Args: [--model <dir>] [--fp16] [--max-batch-tokens N]
    const char* model_dir = "models/qwen2.5-0.5b";
    bool use_fp16 = false;
    int max_batch_tokens = 256;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model") && i+1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--fp16"))                use_fp16  = true;
        else if (!strcmp(argv[i], "--max-batch-tokens") && i+1 < argc) max_batch_tokens = std::atoi(argv[++i]);
    }

    printf("=== NanoInfer Throughput vs Concurrency (EngineServer) ===\n");
    printf("Model: %s | prompt=5 tok | gen=8 tok | %s\n\n",
           model_dir, use_fp16 ? "fp16" : "fp32");

    // One engine; both modes reuse it.  (Engine init is NOT included in timing.)
    EngineServer engine(model_dir, /*max_seq_len=*/0, max_batch_tokens, /*kv_cache_mb=*/0,
                        prefix_cache_policy_from_env(), use_fp16);

    // Sequential baseline prompt (5 tokens).
    std::vector<int> prompt = {576, 8319, 315, 13466, 374};
    const int max_new = 8;
    const float temperature = 0.8f;

    printf("  N     |  Seq tok/s |  Batch tok/s |  Speedup (batch/seq)\n");
    printf("  ------|------------|--------------|---------------------\n");

    int batches[] = {1, 2, 4, 8, 16};

    for (int b = 0; b < 5; b++) {
        int N = batches[b];

        // ---- Sequential: N requests, one at a time ----
        double seq_ms = 0;
        {
            BatchMainLoop loop(engine, SchedulerPolicy::FCFS, 16, max_batch_tokens);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; i++) {
                int id = loop.submit(prompt, max_new, 151643, "", "", temperature);
                loop.run();
                (void)loop.generated_tokens(id).size();
            }
            seq_ms = wall_ms(t0);
        }  // loop destructor releases KV blocks

        // ---- Batched: all N submitted together, continuous batching ----
        double bat_ms = 0;
        {
            BatchMainLoop loop(engine, SchedulerPolicy::DecodeFirst, 16, max_batch_tokens);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; i++)
                loop.submit(prompt, max_new, 151643, "", "", temperature);
            loop.run();
            bat_ms = wall_ms(t0);
        }

        double seq_tok = N * max_new * 1000.0 / seq_ms;
        double bat_tok = N * max_new * 1000.0 / bat_ms;
        printf("  %5d  | %10.1f | %12.1f |  %5.2fx\n",
               N, seq_tok, bat_tok, bat_tok / seq_tok);
        fflush(stdout);
    }

    printf("\nNote: KV pool init + first-request warmup affect small-N timing.\n");
    return 0;
}
