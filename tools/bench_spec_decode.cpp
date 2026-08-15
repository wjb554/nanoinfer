/// bench_spec_decode.cpp — Speculative Decoding Speedup Benchmark
///
/// Two modes:
///   --self [target_dir]          Self-speculation (first N layers as draft)
///   --dual <target_dir> <draft>  Dual-model (separate draft model, free pairing)
///
/// Usage:
///   bench_spec_decode.exe                                        # self-spec default
///   bench_spec_decode.exe --self models/qwen2.5-0.5b             # self-spec
///   bench_spec_decode.exe --dual models/qwen2.5-1.5b models/qwen2.5-0.5b  # dual
///
/// Uses EngineServer + BatchMainLoop for both modes.
/// Temperature=0 (greedy).  3 runs median.  k={1,2,3,4,5,6,8}.

#include "nanoinfer/engine/batch_loop.h"
#include "nanoinfer/engine/engine.h"
#include "nanoinfer/kv_cache/prefix_cache.h"
#include "nanoinfer/tokenizer/tokenizer.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace nanoinfer;
using namespace nanoinfer::engine;

static double median_of_three(double a, double b, double c) {
    if (a > b) std::swap(a, b); if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b); return b;
}

int main(int argc, char** argv) {
    // ---- Parse args ----
    bool dual_mode = false;
    int draft_layers = 6;  // self-speculation depth (fraction of target layers)
    const char* target_dir = "models/qwen2.5-0.5b";
    const char* draft_dir  = "models/qwen2.5-0.5b";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dual")) {
            dual_mode = true;
            if (i+2 < argc) { target_dir = argv[++i]; draft_dir = argv[++i]; }
        } else if (!strcmp(argv[i], "--self")) {
            dual_mode = false;
            if (i+1 < argc) target_dir = argv[++i];
        } else if (!strcmp(argv[i], "--draft-layers") && i+1 < argc) {
            draft_layers = atoi(argv[++i]);
        }
    }

    // ---- Header ----
    printf("=== Speculative Decoding Benchmark ===\n");
    printf("Target: %s\n", target_dir);
    if (dual_mode) printf("Draft:  %s (dual-model)\n", draft_dir);
    else           printf("Draft:  first %d layers (self-speculation)\n", draft_layers);
    printf("GPU: auto (detected at runtime)\n\n");

    // ---- Load target model ----
    // max_seq_len=0 -> use config default + auto-size the KV pool from GPU
    // memory.  (The old hardcoded 128 starved the pool to 16 blocks/layer,
    // which exhausted during dual-mode speculative pre-allocation -> garbage
    // target logits -> 0% acceptance.  The draft engine inherits this size.)
    printf("Loading target model...\n");
    EngineServer target(target_dir, /*max_seq_len=*/0, /*max_batch_tokens=*/256, 0,
        nanoinfer::kv_cache::prefix_cache_policy_from_env(),
        /*use_fp16=*/true);

    // ---- Enable speculative decode ONCE (dual-mode loads draft model) ----
    if (dual_mode) {
        SpeculativeDecodeConfig cfg;
        cfg.enabled = true;
        cfg.draft_model_dir = draft_dir;
        // Argmax-equality acceptance collapses to ~0% for cross-scale dual
        // models (0.5B draft rarely matches the 7B target's argmax).  Use
        // Leviathan probability acceptance min(1, p_target/p_draft) instead.
        cfg.probability_acceptance = true;
        // Batch the k-token verify into ONE M=k forward instead of k
        // single-token passes — amortizes the per-layer weight loads that
        // otherwise make spec decode net-negative.
        cfg.verify_batch = true;
        target.enable_speculative_decode(cfg);
    }

    // ---- Tokenizer (draft and target share same vocab for Qwen2.5) ----
    char tok_path[512];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", target_dir);
    nanoinfer::tokenizer::Tokenizer tok(tok_path);

    // ---- Prompts ----
    struct { const char* text; const char* label; } prompts[] = {
        {"The capital", "short (~5 tok)"},
        {"The capital of France is a beautiful city known for its", "medium (~15 tok)"},
        {"The capital of France is a beautiful city known for its rich history and cultural heritage", "long (~20 tok)"},
    };
    int k_vals[] = {1,2,3,4,5,6,8}, nk = 7;
    int max_new_vals[] = {16, 32, 64};
    int eos = 151643;

    // ---- Run ----
    for (auto& p : prompts) {
        auto ptoks = tok.encode(p.text);
        printf("Prompt: \"%s\" -> %d tokens [%s]\n\n", p.text, (int)ptoks.size(), p.label);

        for (int mn : max_new_vals) {
            printf("=== max_new=%d ===\n", mn);
            printf("%-6s %12s %12s %10s %8s\n", "k","regular(s)","spec(s)","speedup","accept");
            printf("%-6s %12s %12s %10s %8s\n", "------","----------","----------","--------","--------");

            for (int ki = 0; ki < nk; ki++) {
                int k = k_vals[ki];
                double reg_t[3], spec_t[3];

                // Regular decode: 3 runs (spec disabled)
                target.set_spec_enabled(false);
                for (int r = 0; r < 3; r++) {
                    target.clear_kv_cache();
                    BatchMainLoop loop(target, SchedulerPolicy::FCFS, 16, 256);
                    int id = loop.submit(ptoks, mn, eos);
                    auto t0 = std::chrono::steady_clock::now();
                    loop.run();
                    auto t1 = std::chrono::steady_clock::now();
                    reg_t[r] = std::chrono::duration<double,std::milli>(t1-t0).count();
                }

                // Speculative decode: 3 runs (spec enabled)
                target.set_spec_enabled(true);
                int total_drafted = 0, total_accepted = 0;
                for (int r = 0; r < 3; r++) {
                    target.clear_kv_cache();
                    BatchMainLoop loop(target, SchedulerPolicy::FCFS, 16, 256);
                    if (!dual_mode) {
                        // DIAGNOSTIC: self-spec with probability acceptance but
                        // SEQUENTIAL verify (no batched-verify numeric
                        // divergence).  Isolates whether the self+prob 0% came
                        // from the divergence or a probability-path bug.
                        loop.enable_speculative_decode(k, "", draft_layers,
                                                       /*verify_batch=*/false,
                                                       /*probability_acceptance=*/true);
                    } else {
                        target.set_num_draft_tokens(k);     // dual: just update k
                    }
                    int id = loop.submit(ptoks, mn, eos);
                    auto t0 = std::chrono::steady_clock::now();
                    loop.run();
                    auto t1 = std::chrono::steady_clock::now();
                    spec_t[r] = std::chrono::duration<double,std::milli>(t1-t0).count();
                    int sd = loop.spec_drafted(id);
                    int sa = loop.spec_accepted(id);
                    total_drafted += sd;
                    total_accepted += sa;
                }

                double rm = median_of_three(reg_t[0],reg_t[1],reg_t[2]);
                double sm = median_of_three(spec_t[0],spec_t[1],spec_t[2]);
                double speedup = rm / sm;
                double accept = total_drafted>0 ? 100.0*total_accepted/total_drafted : 0;

                printf("k=%-4d %8.2fs    %8.2fs    %6.2fx    %5.0f%%\n",
                       k, rm/1000, sm/1000, speedup, accept);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("Done.\n");
    return 0;
}
