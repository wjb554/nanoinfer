/// bench_eval.cpp — batch evaluation on test sets (runs on the server).
///
/// Modes:
///   --mode accept    : per-prompt speculative acceptance + speedup (regular 7B vs 7B+0.5B dual)
///   --mode accuracy  : per-prompt greedy output vs reference answer (gsm8k number / mmlu letter)
/// JSONL input: {"prompt": "...", "answer": "..."}  (answer used only in accuracy mode)
///
/// Usage:
///   bench_eval --mode accept    --data bench_data/alpaca_sample.jsonl --sample 100 --model models/qwen2.5-7b --draft models/qwen2.5-0.5b --fp16
///   bench_eval --mode accuracy  --data bench_data/gsm8k_test.jsonl   --sample 200 --model models/qwen2.5-7b --fp16
///   bench_eval --mode accuracy  --data bench_data/mmlu_test_sample.jsonl --sample 200 --model models/qwen2.5-7b --fp16

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "nanoinfer/engine/engine.h"
#include "nanoinfer/engine/batch_loop.h"
#include "nanoinfer/tokenizer/tokenizer.h"
#include "picojson.h"

using namespace nanoinfer::engine;

struct EvalRow {
    std::string prompt, answer;
    std::vector<int> tokens;  // pre-tokenized (HF) if present, else empty -> use simplified encode
};

static std::vector<EvalRow> load_jsonl(const std::string& path) {
    std::vector<EvalRow> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        picojson::value v;
        std::string err = picojson::parse(v, line);
        if (!err.empty()) continue;
        if (!v.is<picojson::object>()) continue;
        picojson::object o = v.get<picojson::object>();
        EvalRow r;
        if (o.count("prompt")) r.prompt = o["prompt"].get<std::string>();
        if (o.count("answer")) r.answer = o["answer"].get<std::string>();
        if (o.count("tokens") && o["tokens"].is<picojson::array>()) {
            for (auto& tv : o["tokens"].get<picojson::array>()) {
                if (tv.is<int64_t>()) r.tokens.push_back((int)tv.get<int64_t>());
                else if (tv.is<double>()) r.tokens.push_back((int)tv.get<double>());
            }
        }
        if (!r.prompt.empty() || !r.tokens.empty()) out.push_back(std::move(r));
    }
    return out;
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)std::min((double)v.size() - 1, std::floor(p * (double)v.size()));
    return v[idx];
}

// gsm8k: answer = CoT ... "#### <number>". Extract the number after the last "####".
static std::string gsm8k_expected(const std::string& answer) {
    size_t pos = answer.rfind("####");
    if (pos == std::string::npos) return "";
    std::string tail = answer.substr(pos + 4);
    std::string num;
    for (char c : tail) { if (isdigit((unsigned char)c)) num += c; else if (!num.empty()) break; }
    return num;
}

// Last number appearing in text (for matching gsm8k answers).
static std::string last_number(const std::string& text) {
    std::string cur, last;
    for (char c : text) {
        if (isdigit((unsigned char)c)) cur += c;
        else { if (!cur.empty()) { last = cur; cur.clear(); } }
    }
    if (!cur.empty()) last = cur;
    return last;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

int main(int argc, char** argv) {
    std::string model_dir = "models/qwen2.5-7b";
    std::string draft_dir = "models/qwen2.5-0.5b";
    std::string data_path, mode = "accept";
    int sample = 100, max_new = 32;
    bool use_fp16 = true;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model") && i + 1 < argc)  model_dir = argv[++i];
        else if (!strcmp(argv[i], "--draft") && i + 1 < argc)  draft_dir = argv[++i];
        else if (!strcmp(argv[i], "--data") && i + 1 < argc)   data_path = argv[++i];
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc)   mode = argv[++i];
        else if (!strcmp(argv[i], "--sample") && i+1 < argc)   sample = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-new") && i+1 < argc)  max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fp16"))                   use_fp16 = true;
        else if (!strcmp(argv[i], "--fp32"))                   use_fp16 = false;
    }
    if (data_path.empty()) { fprintf(stderr, "usage: bench_eval --mode accept|accuracy --data <jsonl> [--model ..] [--draft ..] [--sample N] [--fp16]\n"); return 1; }

    auto rows = load_jsonl(data_path);
    printf("== bench_eval mode=%s data=%s rows=%zu sample=%d ==\n", mode.c_str(), data_path.c_str(), rows.size(), sample);
    if (rows.size() > (size_t)sample) {
        std::mt19937 rng(42);
        std::shuffle(rows.begin(), rows.end(), rng);
        rows.resize(sample);
    }

    EngineServer engine(model_dir, /*max_seq_len=*/0, /*max_batch_tokens=*/256, /*kv_cache_mb=*/0,
                        nanoinfer::kv_cache::prefix_cache_policy_from_env(), use_fp16);
    nanoinfer::tokenizer::Tokenizer tok(model_dir + "/tokenizer.json");
    printf("[init] model loaded. layers=%d hidden=%d vocab=%d\n", engine.num_layers(), engine.hidden_dim(), engine.vocab_size());

    bool accept_mode = (mode == "accept");
    if (accept_mode) {
        SpeculativeDecodeConfig cfg;
        cfg.enabled = true;
        cfg.draft_model_dir = draft_dir;
        cfg.num_draft_tokens = 4;
        cfg.verify_batch = true;
        engine.enable_speculative_decode(cfg);
        printf("[init] spec decode enabled (dual %s, k=4, verify_batch)\n", draft_dir.c_str());
    }

    std::vector<double> acceptances, speedups, reg_tpots, spec_tpots;
    int total = 0, ident = 0, correct = 0, skipped_tok = 0;
    auto t_start = std::chrono::steady_clock::now();

    for (auto& row : rows) {
        auto ptoks = row.tokens.empty() ? tok.encode(row.prompt) : row.tokens;
        if (ptoks.empty()) { skipped_tok++; continue; }
        const std::string& answer = row.answer;

        // ---- regular greedy ----
        engine.set_spec_enabled(false);
        engine.clear_kv_cache();
        std::string reg_text; double reg_tpot = 0;
        { BatchMainLoop loop(engine, SchedulerPolicy::FCFS, 16, 256);
          int id = loop.submit(ptoks, max_new, 151643);
          loop.run();
          reg_tpot = loop.metrics(id).tpot_ms();
          reg_text = tok.decode(loop.generated_tokens(id)); }
        engine.clear_kv_cache();

        if (!accept_mode) {
            // ---- accuracy mode ----
            total++;
            bool ok = false;
            if (data_path.find("gsm8k") != std::string::npos) {
                std::string exp = gsm8k_expected(answer);
                if (!exp.empty()) ok = (last_number(reg_text) == exp);
            } else if (data_path.find("mmlu") != std::string::npos) {
                std::string t = trim(reg_text);
                std::string exp = trim(answer);
                if (!t.empty() && !exp.empty()) {
                    // first character match (A/B/C/D) OR the letter appears near the start
                    ok = (t[0] == exp[0]) || (t.find(exp[0]) != std::string::npos && t.find(exp[0]) <= 4);
                }
            } else {
                // generic: exact/containment of trimmed answer
                ok = (trim(reg_text) == trim(answer)) || (trim(answer).size() > 3 && reg_text.find(trim(answer)) != std::string::npos);
            }
            if (ok) correct++;
            printf("[%4d] %s\n", total, ok ? "OK " : "NO ");
            continue;
        }

        // ---- spec mode: 7B+0.5B dual ----
        engine.set_spec_enabled(true);
        std::string spec_text; double spec_tpot = 0, acc = 0;
        { BatchMainLoop loop(engine, SchedulerPolicy::FCFS, 16, 256);
          int id = loop.submit(ptoks, max_new, 151643);
          loop.run();
          spec_tpot = loop.metrics(id).tpot_ms();
          auto ss = loop.spec_stats(id);
          acc = (ss.accepted + ss.rejected) > 0 ? 100.0 * ss.accepted / (ss.accepted + ss.rejected) : 0.0;
          spec_text = tok.decode(loop.generated_tokens(id)); }
        engine.clear_kv_cache();

        acceptances.push_back(acc);
        reg_tpots.push_back(reg_tpot);
        spec_tpots.push_back(spec_tpot);
        speedups.push_back(reg_tpot / std::max(spec_tpot, 1e-9));
        total++;
        if (reg_text == spec_text) ident++;
    }

    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("\n==== SUMMARY ====\n");
    printf("total=%d skipped_tokenize=%d elapsed=%.1fs\n", total, skipped_tok, elapsed);
    if (accept_mode) {
        printf("acceptance: mean=%.1f%% p50=%.1f%% p90=%.1f%% min=%.1f%% max=%.1f%%\n",
               acceptances.empty()?0:std::accumulate(acceptances.begin(),acceptances.end(),0.0)/acceptances.size(),
               percentile(acceptances,0.5), percentile(acceptances,0.9),
               acceptances.empty()?0:*std::min_element(acceptances.begin(),acceptances.end()),
               acceptances.empty()?0:*std::max_element(acceptances.begin(),acceptances.end()));
        printf("speedup  : mean=%.2fx p50=%.2fx p90=%.2fx | #speedup>1.0 = %zu/%zu\n",
               speedups.empty()?0:std::accumulate(speedups.begin(),speedups.end(),0.0)/speedups.size(),
               percentile(speedups,0.5), percentile(speedups,0.9),
               std::count_if(speedups.begin(),speedups.end(),[](double s){return s>1.0;}),
               speedups.size());
        printf("reg TPOT : mean=%.1fms p50=%.1fms | spec TPOT: mean=%.1fms p50=%.1fms\n",
               reg_tpots.empty()?0:std::accumulate(reg_tpots.begin(),reg_tpots.end(),0.0)/reg_tpots.size(),
               percentile(reg_tpots,0.5),
               spec_tpots.empty()?0:std::accumulate(spec_tpots.begin(),spec_tpots.end(),0.0)/spec_tpots.size(),
               percentile(spec_tpots,0.5));
        printf("spec==regular text identical: %d/%d (%.1f%%)\n", ident, total, total?100.0*ident/total:0.0);
    } else {
        printf("accuracy: %d/%d (%.1f%%)\n", correct, total, total?100.0*correct/total:0.0);
    }
    return 0;
}
