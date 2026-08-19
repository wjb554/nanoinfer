/// quick_generate — minimal text→text sanity test (EngineServer + BatchMainLoop).
/// Feeds a text prompt, generates tokens, decodes back to text and prints it.
///
/// Usage:
///   quick_generate [--model <dir>] [--fp16] [--max-new N] [--prompt <text>]
///   quick_generate --spec K [--draft <dir>] [--draft-layers N]
///                  [--verify-batch] [--prob]   # speculative decoding
///
/// Example:
///   quick_generate --model models/qwen2.5-7b --fp16 \
///       --prompt "The capital of France is"
///   quick_generate --model models/qwen2.5-7b --fp16 --spec 4 \
///       --prompt "The capital of France is"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "nanoinfer/engine/engine.h"
#include "nanoinfer/engine/batch_loop.h"
#include "nanoinfer/tokenizer/tokenizer.h"
#include "bench_common.h"

using namespace nanoinfer::engine;

int main(int argc, char** argv) {
    const char* model_dir = "models/qwen2.5-0.5b";
    bool use_fp16 = true;   // FP16 default; pass --fp32 for exact-FP32 repro
    bool use_fp8  = false;   // FP8 weight-only (E4M3)
    int  max_new = 32;
    float temperature = 1.0f;
    int  max_batch_tokens = 0;
    int  single_token = -1;
    std::string prompt = "The capital of France is";
    std::string tokens_arg;

    // Speculative decoding options.
    int k_spec = 0;              // draft tokens per step (0 = disabled)
    std::string draft_dir;       // draft model dir (empty = self-speculation)
    int draft_layers = 0;        // self-speculation depth M (0 = auto)
    bool verify_batch = false;   // batched (M=k) verify vs sequential (M=1)
    bool prob_accept = false;    // Leviathan probability acceptance

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model") && i+1 < argc)   model_dir = argv[++i];
        else if (!strcmp(argv[i], "--fp16"))                  use_fp16 = true;
        else if (!strcmp(argv[i], "--fp32"))                  use_fp16 = false;
        else if (!strcmp(argv[i], "--fp8"))                   use_fp8  = true;
        else if (!strcmp(argv[i], "--max-new") && i+1<argc)   max_new = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc)  prompt = argv[++i];
        else if (!strcmp(argv[i], "--token") && i+1 < argc)   single_token = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i+1 < argc)  tokens_arg = argv[++i];
        else if (!strcmp(argv[i], "--temp") && i+1 < argc)    temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-batch-tokens") && i+1<argc) max_batch_tokens = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--spec") && i+1 < argc)           k_spec = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--draft") && i+1 < argc)          draft_dir = argv[++i];
        else if (!strcmp(argv[i], "--draft-layers") && i+1 < argc)   draft_layers = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify-batch"))                 verify_batch = true;
        else if (!strcmp(argv[i], "--prob"))                         prob_accept = true;
    }
    if (max_batch_tokens <= 0) max_batch_tokens = default_batch_tokens();

    EngineServer engine(model_dir, /*max_seq_len=*/0, max_batch_tokens, 0,
                        nanoinfer::kv_cache::prefix_cache_policy_from_env(),
                        use_fp16, use_fp8);
    nanoinfer::tokenizer::Tokenizer tok(std::string(model_dir) + "/tokenizer.json");

    std::vector<int> ptoks;
    if (!tokens_arg.empty()) {
        size_t pos = 0;
        while (pos < tokens_arg.size()) {
            size_t c = tokens_arg.find(',', pos);
            ptoks.push_back(std::atoi(tokens_arg.substr(pos, c - pos).c_str()));
            if (c == std::string::npos) break;
            pos = c + 1;
        }
    } else if (single_token >= 0) { ptoks.push_back(single_token); }
    else { ptoks = tok.encode(prompt); }
    printf("[prompt] %d tokens:", (int)ptoks.size());
    for (int t : ptoks) printf(" %d", t);
    printf("\n");

    BatchMainLoop loop(engine, SchedulerPolicy::FCFS, 16, max_batch_tokens);
    if (k_spec > 0)
        loop.enable_speculative_decode(k_spec, draft_dir, draft_layers,
                                       verify_batch, prob_accept);
    int id = loop.submit(ptoks, max_new, 151643, "", "", temperature);
    loop.run();

    if (k_spec > 0) {
        SpeculativeStats ss = loop.spec_stats(id);
        printf("[SPEC] drafted=%d accepted=%d rejected=%d steps=%d\n",
               ss.drafted, ss.accepted, ss.rejected, ss.steps);
        double acc = (ss.accepted + ss.rejected) > 0
                     ? 100.0 * ss.accepted / (ss.accepted + ss.rejected)
                     : 0.0;
        printf("[SPEC] acceptance=%.1f%%\n", acc);
        printf("[SPEC] draft_ms=%.1f verify_ms=%.1f total_ms=%.1f\n",
               ss.draft_ms, ss.verify_ms, ss.total_ms);
    }

    std::vector<int> out = loop.generated_tokens(id);
    std::string text = tok.decode(out);
    printf("[PROMPT] %s\n", prompt.c_str());
    printf("[TOKENS]");
    for (int t : out) printf(" %d", t);
    printf("\n");
    printf("[OUTPUT] %s\n", text.c_str());
    printf("[stats] %d tokens, %.1f ms, TTFT=%.1f ms, TPOT=%.1f ms\n",
           (int)out.size(), loop.metrics(id).latency_ms(),
           loop.metrics(id).ttft_ms(), loop.metrics(id).tpot_ms());
    return 0;
}
