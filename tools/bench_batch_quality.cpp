/// bench_batch_quality — run a FIXED set of N requests (each prompt 50-200
/// tokens) through the step() batch path (BatchMainLoop), for fp16 vs fp8
/// output comparison across models.
///
/// The prompt set is built deterministically from an embedded corpus file
/// (one paragraph per line), so every run — fp16 / fp8 / any model — gets the
/// exact same inputs.  Each prompt starts at a paragraph boundary and is
/// filled to a length in [min_len, max_len] by cycling through paragraphs.
///
/// Usage:
///   bench_batch_quality [--model <dir>] [--fp8] [--corpus <file>]
///                       [--num-req N] [--min-len L] [--max-len L] [--max-new M]
///   (defaults: 100 requests, prompt 50-200 tokens, greedy, 32 output tokens)

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>

#include "nanoinfer/engine/engine.h"
#include "nanoinfer/engine/batch_loop.h"
#include "nanoinfer/tokenizer/tokenizer.h"
#include "bench_common.h"

using namespace nanoinfer;
using namespace nanoinfer::engine;

int main(int argc, char** argv) {
    const char* model_dir   = "models/qwen2.5-0.5b";
    const char* corpus_file = "tools/prompts_corpus.txt";
    bool use_fp8 = false;
    int num_req = 100, min_len = 50, max_len = 200, max_new = 32;
    int max_batch_tokens = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model") && i+1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--corpus") && i+1 < argc) corpus_file = argv[++i];
        else if (!strcmp(argv[i], "--fp8"))                  use_fp8  = true;
        else if (!strcmp(argv[i], "--num-req") && i+1<argc)  num_req  = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-len") && i+1<argc)  min_len  = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-len") && i+1<argc)  max_len  = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-new") && i+1<argc)  max_new  = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-batch-tokens") && i+1<argc)
            max_batch_tokens = std::atoi(argv[++i]);
    }
    if (max_batch_tokens <= 0) max_batch_tokens = default_batch_tokens();

    // ---- Read corpus + tokenize per paragraph ----
    tokenizer::Tokenizer tok(std::string(model_dir) + "/tokenizer.json");
    std::ifstream f(corpus_file);
    std::string line;
    std::vector<std::vector<int>> paras;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<int> t = tok.encode(line);
        if (!t.empty()) paras.push_back(std::move(t));
    }
    if (paras.empty()) { fprintf(stderr, "corpus empty\n"); return 2; }
    int P = (int)paras.size();

    // ---- Build deterministic prompt set (identical every run) ----
    std::vector<std::vector<int>> prompts(num_req);
    std::vector<long long> psum(num_req, 0);
    for (int i = 0; i < num_req; i++) {
        int L = min_len + (int)((long long)(max_len - min_len) * i
                                / std::max(1, num_req - 1));
        int p = i % P;                 // start at a paragraph boundary
        size_t pos = 0;
        prompts[i].reserve(L);
        for (int t = 0; t < L; t++) {
            while (pos >= paras[p].size()) { pos = 0; p = (p + 1) % P; }
            prompts[i].push_back(paras[p][pos++]);
            psum[i] += prompts[i].back();
        }
    }

    // ---- Engine + batch through step() ----
    EngineServer engine(model_dir, 0, max_batch_tokens, 0,
                        kv_cache::prefix_cache_policy_from_env(),
                        /*use_fp16=*/true, use_fp8);
    BatchMainLoop batch(engine, SchedulerPolicy::DecodeFirst, 16,
                        max_batch_tokens);

    printf("# mode=%s model=%s num_req=%d min_len=%d max_len=%d max_new=%d batch=%d\n",
           use_fp8 ? "fp8" : "fp16", model_dir, num_req, min_len, max_len,
           max_new, max_batch_tokens);
    fflush(stdout);
    for (int i = 0; i < num_req; i++)
        batch.submit(prompts[i], max_new, 151643, "", "", 0.0f);  // greedy
    batch.run();

    // ---- Per-request report: prompt checksum + generated tokens ----
    for (int i = 0; i < num_req; i++) {
        const auto& out = batch.generated_tokens(i);
        const auto& m = batch.metrics(i);
        printf("[REQ %d] psum=%lld plen=%d ttft=%.1f tpot=%.1f tok=",
               i, psum[i], (int)prompts[i].size(), m.ttft_ms(), m.tpot_ms());
        for (int t : out) printf(" %d", t);
        printf("\n");
    }
    printf("# total_tokens=%d throughput=%.1f tok/s\n",
           (int)batch.total_tokens_generated(), batch.throughput_tok_s());
    return 0;
}
