/// bench_prompt_lengths.cpp — single-request latency vs prompt length.
///
/// Builds prompts at target token lengths (50..500), runs each through
/// EngineServer + BatchMainLoop (one request at a time), and reports the
/// standard latency metrics (TTFT / TPOT / total / throughput) PLUS the
/// actual input and output TEXT (written to stdout and a log file).
///
/// Usage:
///   bench_prompt_lengths [--model <dir>] [--fp16] [--max-new N]
///                        [--lengths a,b,c,...] [--max-batch-tokens N]
///                        [--out <logfile>]
///
/// Example:
///   bench_prompt_lengths --model models/qwen2.5-7b --fp16 --max-new 100

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "lightllm/engine/engine.h"
#include "lightllm/engine/batch_loop.h"
#include "lightllm/tokenizer/tokenizer.h"
#include "bench_common.h"

using namespace lightllm::engine;
using namespace lightllm::kv_cache;

// ---- Build a long, VARIED English corpus.  Repeating a single paragraph makes
// the model degrade (off-distribution repetition); a varied multi-topic passage
// keeps generated output meaningful so the recorded text is useful.  We feed
// token IDs directly, so the prompt length is EXACT regardless of the C++
// tokenizer's over-splitting.
static const char* kParagraphs[] = {
    "The capital of France is Paris, a city known for its rich history, art, and culture. "
    "It is one of the most visited cities in the world, famous for landmarks such as the "
    "Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. ",
    "The city sits on the River Seine and has been an important center of learning and "
    "innovation for centuries, home to universities, libraries, and world-class museums. ",
    "Paris is also a global hub for fashion, cuisine, and technology, attracting millions "
    "of visitors every year who explore its cafes, galleries, and historic boulevards. ",
    "The French capital blends old and new, where medieval streets meet modern "
    "architecture and where tradition coexists with a vibrant start-up scene. ",
    "France itself has a long and influential history, from the monarchy to the republic, "
    "and its language and culture have spread around the world. ",
    "Beyond Paris, the country is famous for its wine regions, alpine peaks, and "
    "Mediterranean coastline, each with its own traditions and regional cuisine. ",
    "In science, French researchers have made discoveries in physics, chemistry, and "
    "mathematics that shaped modern thought and technology. ",
    "The country also played a key role in European politics and the arts, producing "
    "painters, writers, and composers whose work is still celebrated today. ",
};
static const int kNumParagraphs =
    static_cast<int>(sizeof(kParagraphs) / sizeof(kParagraphs[0]));

static std::string build_corpus(int target_tokens) {
    // Cycle through distinct paragraphs (varied content) until the corpus is
    // comfortably longer than the target prompt length.
    std::string corpus;
    for (int i = 0; corpus.size() < (size_t)target_tokens * 4; i++)
        corpus += kParagraphs[i % kNumParagraphs];
    return corpus;
}

int main(int argc, char** argv) {
    const char* model_dir = "models/qwen2.5-7b";
    bool use_fp16 = false;
    int  max_new = 100;
    float temperature = 0.0f;   // greedy (deterministic) by default
    int  max_batch_tokens = 0;
    std::vector<int> lengths = {50, 100, 200, 300, 400, 500};
    std::string out_file = "prompt_lengths.log";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model") && i+1 < argc)           model_dir = argv[++i];
        else if (!strcmp(argv[i], "--fp16"))                          use_fp16 = true;
        else if (!strcmp(argv[i], "--max-new") && i+1 < argc)         max_new = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i+1 < argc)            temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-batch-tokens") && i+1<argc)  max_batch_tokens = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i+1 < argc)             out_file = argv[++i];
        else if (!strcmp(argv[i], "--lengths") && i+1 < argc) {
            lengths.clear();
            const char* s = argv[++i];
            char buf[64]; const char* p = s;
            while (*p) {
                const char* c = strchr(p, ',');
                size_t n = c ? (size_t)(c - p) : strlen(p);
                memcpy(buf, p, n); buf[n] = 0;
                lengths.push_back(atoi(buf));
                if (!c) break; p = c + 1;
            }
        }
    }
    if (max_batch_tokens <= 0) max_batch_tokens = default_batch_tokens();

    // ---- Load engine (fp16) ----
    printf("Loading %s (%s)...\n", model_dir, use_fp16 ? "FP16" : "FP32");
    EngineServer engine(model_dir, /*max_seq_len=*/0, max_batch_tokens, /*kv_cache_mb=*/0,
                        prefix_cache_policy_from_env(), use_fp16);
    lightllm::tokenizer::Tokenizer tok(std::string(model_dir) + "/tokenizer.json");

    // ---- Build the corpus token list (long) ----
    std::string corpus = build_corpus(1000);
    std::vector<int> corpus_toks = tok.encode(corpus);
    printf("Corpus: %d tokens available\n", (int)corpus_toks.size());

    FILE* logf = fopen(out_file.c_str(), "w");
    if (!logf) { fprintf(stderr, "cannot open %s\n", out_file.c_str()); return 1; }

    fprintf(logf, "# LightLLM prompt-length latency report\n");
    fprintf(logf, "# model=%s fp16=%d max_new=%d\n", model_dir, use_fp16 ? 1 : 0, max_new);
    fprintf(logf, "# GPU auto (server RTX 4090)\n\n");

    printf("=== Prompt-Length Latency (%d output tokens each) ===\n", max_new);
    printf("%-8s %10s %10s %10s %10s %10s\n",
           "prompt", "TTFT(ms)", "TPOT(ms)", "total(ms)", "gen_tok/s", "e2e_tok/s");
    printf("%-8s %10s %10s %10s %10s %10s\n",
           "-------", "--------", "--------", "--------", "--------", "--------");

    for (int L : lengths) {
        if (L > (int)corpus_toks.size()) {
            fprintf(stderr, "skip L=%d (corpus too short)\n", L);
            continue;
        }
        std::vector<int> prompt(corpus_toks.begin(), corpus_toks.begin() + L);
        std::string in_text = tok.decode(prompt);

        // One request through BatchMainLoop.
        BatchMainLoop loop(engine, SchedulerPolicy::FCFS, /*chunk_size=*/16, max_batch_tokens);
        int id = loop.submit(prompt, max_new, /*eos=*/151643, "", "", temperature);
        loop.run();

        auto gen = loop.generated_tokens(id);
        auto& m = loop.metrics(id);
        std::string out_text = tok.decode(gen);

        double gen_tok_s = (m.output_len > 0 && m.tpot_ms() > 0)
            ? 1000.0 / m.tpot_ms() : 0;
        double e2e_tok_s = (m.output_len > 0 && m.latency_ms() > 0)
            ? m.output_len * 1000.0 / m.latency_ms() : 0;

        printf("%-8d %10.1f %10.2f %10.1f %10.1f %10.1f\n",
               L, m.ttft_ms(), m.tpot_ms(), m.latency_ms(), gen_tok_s, e2e_tok_s);
        fflush(stdout);

        fprintf(logf, "\n----- prompt_len=%d (prefill %d tok) -----\n", L, L);
        fprintf(logf, "[INPUT ] %s\n", in_text.c_str());
        fprintf(logf, "[OUTPUT] %s\n", out_text.c_str());
        fprintf(logf, "TTFT=%.1f ms  TPOT=%.2f ms  total=%.1f ms  "
                      "gen=%.1f tok/s  e2e=%.1f tok/s  out_tokens=%d\n",
                m.ttft_ms(), m.tpot_ms(), m.latency_ms(),
                gen_tok_s, e2e_tok_s, m.output_len);
    }

    fclose(logf);
    printf("\nInput/output text saved to %s\n", out_file.c_str());
    return 0;
}
