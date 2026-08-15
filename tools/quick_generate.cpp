/// quick_generate — minimal text→text sanity test (EngineServer + BatchMainLoop).
/// Feeds a text prompt, generates tokens, decodes back to text and prints it.
///
/// Usage:
///   quick_generate [--model <dir>] [--fp16] [--max-new N] [--prompt <text>]
///
/// Example:
///   quick_generate --model models/qwen2.5-7b --fp16 \
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
    bool use_fp16 = false;
    int  max_new = 32;
    float temperature = 1.0f;
    int  max_batch_tokens = 0;
    int  single_token = -1;
    std::string prompt = "The capital of France is";
    std::string tokens_arg;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model") && i+1 < argc)   model_dir = argv[++i];
        else if (!strcmp(argv[i], "--fp16"))                  use_fp16 = true;
        else if (!strcmp(argv[i], "--max-new") && i+1<argc)   max_new = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc)  prompt = argv[++i];
        else if (!strcmp(argv[i], "--token") && i+1 < argc)   single_token = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i+1 < argc)  tokens_arg = argv[++i];
        else if (!strcmp(argv[i], "--temp") && i+1 < argc)    temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-batch-tokens") && i+1<argc) max_batch_tokens = std::atoi(argv[++i]);
    }
    if (max_batch_tokens <= 0) max_batch_tokens = default_batch_tokens();

    EngineServer engine(model_dir, /*max_seq_len=*/0, max_batch_tokens, 0,
                        nanoinfer::kv_cache::prefix_cache_policy_from_env(),
                        use_fp16);
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
    int id = loop.submit(ptoks, max_new, 151643, "", "", temperature);
    loop.run();

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
