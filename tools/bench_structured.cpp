/// bench_structured.cpp — Constrained (JSON-schema) decoding overhead benchmark.
///
/// Runs the SAME prompt twice per iteration: once unconstrained, once with a
/// JSON-schema constraint (xgrammar FSM).  Reports wall time, tokens/s,
/// TTFT/TPOT and the overhead the constraint adds, and validates the
/// constrained output is syntactically JSON.
///
/// Usage:
///   bench_structured [--model <dir>] [--fp16] [--max-batch-tokens N]
///                    [--kv-cache-mb N] [--max-new N=32] [--runs N=3]
///                    [--schema <json>] [--prompt <text>]
///
/// Example:
///   bench_structured --model models/qwen2.5-1.5b --fp16

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nanoinfer/engine/engine.h"
#include "nanoinfer/engine/batch_loop.h"
#include "nanoinfer/tokenizer/tokenizer.h"
#include "bench_common.h"

using namespace nanoinfer::engine;

static double median(std::vector<double>& v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Lightweight JSON-object validity: first non-space is '{', last non-space '}',
// braces balanced, and a mandatory colon present.  (Full schema conformance is
// validated by tests/test_structured_e2e.cpp.)
static bool looks_like_json(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    size_t e = s.find_last_not_of(" \t\r\n");
    if (b == std::string::npos || e == std::string::npos) return false;
    if (s[b] != '{' || s[e] != '}') return false;
    int depth = 0;
    bool in_str = false;
    for (size_t i = b; i <= e; i++) {
        char c = s[i];
        if (in_str) {
            if (c == '"' && (i == 0 || s[i - 1] != '\\')) in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '{') depth++;
        else if (c == '}') { if (--depth < 0) return false; }
    }
    return depth == 0 && s.find(':') != std::string::npos;
}

int main(int argc, char** argv) {
    const char* model_dir = "models/qwen2.5-0.5b";
    bool use_fp16 = false;
    int  max_batch_tokens = 0, kv_cache_mb = 0, max_new = 64, runs = 3;
    double temp = 0.0;   // greedy by default (structured output is usually greedy)
    std::string schema =
        R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]})";
    std::string prompt_text =
        "Alice is a 30-year-old engineer. Output her info as a JSON object "
        "with \"name\" and \"age\" keys. Only output the JSON:";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model") && i+1 < argc)          model_dir = argv[++i];
        else if (!strcmp(argv[i], "--fp16"))                         use_fp16 = true;
        else if (!strcmp(argv[i], "--max-batch-tokens") && i+1<argc) max_batch_tokens = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-cache-mb") && i+1 < argc)    kv_cache_mb = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-new") && i+1 < argc)        max_new = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--runs") && i+1 < argc)           runs = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i+1 < argc)           temp = std::atof(argv[++i]);
        else if (!strcmp(argv[i], "--schema") && i+1 < argc)         schema = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc)         prompt_text = argv[++i];
    }
    if (max_batch_tokens <= 0) max_batch_tokens = default_batch_tokens();

    EngineServer engine(model_dir, /*max_seq_len=*/0, max_batch_tokens, kv_cache_mb,
                        nanoinfer::kv_cache::prefix_cache_policy_from_env(),
                        use_fp16);

    std::string tok_path = std::string(model_dir) + "/tokenizer.json";
    nanoinfer::tokenizer::Tokenizer tok(tok_path);

    std::vector<int> prompt = tok.encode(prompt_text);
    int eos = 151643;

    printf("=== Structured Output Overhead Benchmark ===\n");
    printf("Model: %s | Precision: %s | max_new=%d | runs=%d | temp=%.1f | batch=%d | kvmb=%d\n\n",
           model_dir, use_fp16 ? "FP16" : "FP32", max_new, runs, temp,
           max_batch_tokens, kv_cache_mb);
    printf("Prompt (%d tok): \"%s\"\n", (int)prompt.size(), prompt_text.c_str());
    printf("Schema: %s\n\n", schema.c_str());
    fflush(stdout);

    struct Res {
        double time_ms = 0, tok_s = 0, ttft = 0, tpot = 0;
        int    out_len = 0;
        bool   json_ok = false;
        std::string text;
    };
    Res uc, sc;

    auto run_mode = [&](bool constrained, Res& out) {
        std::vector<double> times, ttfts, tpots;
        std::vector<int> outlens;
        int json_ok_cnt = 0;
        for (int r = 0; r < runs; r++) {
            engine.clear_kv_cache();
            BatchMainLoop loop(engine, SchedulerPolicy::FCFS, 16, max_batch_tokens);
            int id = loop.submit(prompt, max_new, eos,
                                 constrained ? schema : std::string(), "", temp);
            auto t0 = std::chrono::steady_clock::now();
            loop.run();
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const auto& m = loop.metrics(id);
            times.push_back(ms);
            ttfts.push_back(m.ttft_ms());
            tpots.push_back(m.tpot_ms());
            outlens.push_back(m.output_len);
            if (constrained) {
                out.text = tok.decode(loop.generated_tokens(id));
                if (looks_like_json(out.text)) json_ok_cnt++;
            }
        }
        out.time_ms = median(times);
        out.ttft    = median(ttfts);
        out.tpot    = median(tpots);
        std::sort(outlens.begin(), outlens.end());
        out.out_len = outlens.empty() ? 0 : outlens[outlens.size() / 2];
        out.tok_s   = out.time_ms > 0 ? out.out_len * 1000.0 / out.time_ms : 0;
        out.json_ok = (json_ok_cnt == runs);
    };

    run_mode(false, uc);
    run_mode(true, sc);

    printf("%-14s %10s %8s %8s %8s %8s\n", "mode", "time(ms)", "tok/s", "TTFT", "TPOT", "out_len");
    printf("%-14s %10.1f %8.1f %8.1f %8.1f %8d\n",
           "unconstrained", uc.time_ms, uc.tok_s, uc.ttft, uc.tpot, uc.out_len);
    printf("%-14s %10.1f %8.1f %8.1f %8.1f %8d\n",
           "constrained", sc.time_ms, sc.tok_s, sc.ttft, sc.tpot, sc.out_len);

    printf("\nOverhead:   time +%.1f%%   throughput -%.1f%%\n",
           uc.time_ms > 0 ? 100.0 * (sc.time_ms - uc.time_ms) / uc.time_ms : 0,
           uc.tok_s > 0 ? 100.0 * (uc.tok_s - sc.tok_s) / uc.tok_s : 0);
    printf("Constrained output valid JSON: %s\n", sc.json_ok ? "YES (all runs)" : "NO");
    if (!sc.text.empty()) {
        printf("Sample output (%zu chars): %s\n", sc.text.size(), sc.text.c_str());
    }
    return 0;
}
