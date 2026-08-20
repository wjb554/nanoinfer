/// serve — NanoInfer HTTP serving tool (P2 serving layer).
/// Concurrent + streaming inference server backed by ServerEngine.
///
/// Usage:
///   serve [--model <dir>] [--fp16] [--port <p>] [--max-batch-tokens <n>]
///         [--chunk-size <n>] [--max-seq-len <n>]
///
/// Endpoints:
///   GET  /health                -> {"status":"ok"}
///   GET  /metrics               -> Prometheus text format (nanoinfer_*)
///   POST /v1/chat/completions   -> body:
///        {"prompt": str, "max_tokens": int, "temperature": float,
///         "stream": bool, "eos": int}
///        Non-streaming returns {"choices":[{"text": "...", "finish_reason": "stop"}]}.
///        Streaming emits SSE "data: {…}" chunks then "data: [DONE]".

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "picojson.h"
#include "nanoinfer/engine/tuning.h"
#include "nanoinfer/server/http_server.h"
#include "nanoinfer/server/server_engine.h"
#include "nanoinfer/tokenizer/tokenizer.h"

using namespace nanoinfer::server;

// ----------------------------------------------------------------------------
// JSON helpers
// ----------------------------------------------------------------------------
// picojson stores integral JSON numbers as int64_t and reals as double; accept
// either so "max_tokens":32 and "max_tokens":32.0 both work.
static double json_num(const picojson::value& v, double def) {
    if (v.is<double>()) return v.get<double>();
    if (v.is<int64_t>()) return (double)v.get<int64_t>();
    return def;
}

// Escape text for embedding inside a JSON string literal.
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

// ----------------------------------------------------------------------------
// Prometheus text metrics
// ----------------------------------------------------------------------------
static std::string prometheus_metrics(const ServerEngine& engine) {
    const ServeMetrics& m = engine.metrics();
    std::string out;
    auto line = [&](const char* name, const char* type, const char* help,
                    long long v) {
        out += "# HELP " + std::string(name) + " " + help + "\n";
        out += "# TYPE " + std::string(name) + " " + type + "\n";
        out += std::string(name) + " " + std::to_string(v) + "\n";
    };

    line("nanoinfer_total_requests", "counter",
         "Total requests submitted.", m.total_requests.load());
    line("nanoinfer_completed_requests", "counter",
         "Requests that reached EOS / max_tokens.", m.completed_requests.load());
    line("nanoinfer_active_requests", "gauge",
         "Currently in-flight requests.", m.active_requests.load());
    line("nanoinfer_total_tokens", "counter",
         "Total generated tokens.", m.total_tokens.load());
    line("nanoinfer_total_steps", "counter",
         "Total scheduler steps executed.", m.total_steps.load());
    line("nanoinfer_prefill_tokens", "counter",
         "Total prefill tokens processed.", m.total_prefill_tokens.load());
    line("nanoinfer_decode_steps", "counter",
         "Total decode entries.", m.total_decode_steps.load());

    double avg = 0, maxv = 0;
    int n = 0;
    m.recent_ttft(avg, maxv, n);
    out += "# HELP nanoinfer_ttft_ms Time to first token (ms).\n";
    out += "# TYPE nanoinfer_ttft_ms gauge\n";
    out += "nanoinfer_ttft_samples " + std::to_string(n) + "\n";
    out += "nanoinfer_ttft_avg_ms " + std::to_string(avg) + "\n";
    out += "nanoinfer_ttft_max_ms " + std::to_string(maxv) + "\n";

    m.recent_tpot(avg, maxv, n);
    out += "# HELP nanoinfer_tpot_ms Time per output token (ms).\n";
    out += "# TYPE nanoinfer_tpot_ms gauge\n";
    out += "nanoinfer_tpot_samples " + std::to_string(n) + "\n";
    out += "nanoinfer_tpot_avg_ms " + std::to_string(avg) + "\n";
    out += "nanoinfer_tpot_max_ms " + std::to_string(maxv) + "\n";
    return out;
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string model_dir = "models/qwen2.5-0.5b";
    bool use_fp16 = true;   // FP16 default; pass --fp32 for exact-FP32 repro
    int port = 8080;
    int max_batch_tokens = nanoinfer::engine::default_batch_tokens();
    int max_seq_len = 2048;
    int chunk_size = 0;     // 0 = VRAM-derived prefill chunk

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)         model_dir = argv[++i];
        else if (!strcmp(argv[i], "--fp16"))                     use_fp16 = true;
        else if (!strcmp(argv[i], "--fp32"))                     use_fp16 = false;
        else if (!strcmp(argv[i], "--port") && i + 1 < argc)     port = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-batch-tokens") && i + 1 < argc)
            max_batch_tokens = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk-size") && i + 1 < argc)
            chunk_size = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-seq-len") && i + 1 < argc)
            max_seq_len = std::atoi(argv[++i]);
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    printf("=== NanoInfer serving tool ===\n");
    printf("  model:            %s\n", model_dir.c_str());
    printf("  precision:        %s\n", use_fp16 ? "FP16" : "FP32");
    printf("  port:             %d\n", port);
    printf("  max batch tokens: %d\n", max_batch_tokens);
    printf("  chunk size:       %d\n", chunk_size > 0 ? chunk_size
                                   : nanoinfer::engine::default_chunk_size());
    printf("  max seq len:      %d\n", max_seq_len);
    fflush(stdout);

    nanoinfer::tokenizer::Tokenizer tokenizer(model_dir + "/tokenizer.json");

    ServerEngine engine(model_dir, max_seq_len, max_batch_tokens, use_fp16,
                        chunk_size);
    engine.start();
    printf("  vocab:            %d\n", engine.vocab_size());
    printf("=====================================\n");
    fflush(stdout);

    run_server(port, [&](const HttpRequest& req, HttpResponseWriter& w) {
        if (req.method == "GET" && req.path == "/health") {
            w.send(200, "application/json", "{\"status\":\"ok\"}");
            return;
        }

        if (req.method == "GET" && req.path == "/metrics") {
            w.send(200, "text/plain; version=0.0.4",
                   prometheus_metrics(engine));
            return;
        }

        if (req.method == "POST" && req.path == "/v1/chat/completions") {
            // ---- parse the JSON body ----
            picojson::value v;
            std::string err;
            picojson::parse(v, req.body.begin(), req.body.end(), &err);
            if (!err.empty() || !v.is<picojson::object>()) {
                w.send(400, "application/json",
                       "{\"error\":\"invalid JSON body\"}");
                return;
            }

            std::string prompt =
                v.get("prompt").is<std::string>()
                    ? v.get("prompt").get<std::string>()
                    : "";
            int   max_tokens = (int)json_num(v.get("max_tokens"), 32.0);
            float temperature = (float)json_num(v.get("temperature"), 1.0);
            int   eos = (int)json_num(v.get("eos"), 151643.0);
            bool  stream =
                v.get("stream").is<bool>() ? v.get("stream").get<bool>() : false;

            // ---- tokenize + submit ----
            std::vector<int> ptoks = tokenizer.encode(prompt);
            if (ptoks.empty()) {
                // A zero-token request is dropped by the DecodeFirst scheduler
                // (prefill_remaining()==0 never enters DECODING), so it would
                // never finish and the client would block forever.
                w.send(400, "application/json",
                       "{\"error\":\"prompt tokenized to zero tokens\"}");
                return;
            }

            ServeRequest sr;
            sr.prompt_tokens = ptoks;
            sr.max_new_tokens = max_tokens;
            sr.eos_token_id = eos;
            sr.temperature = temperature;
            int id = engine.submit(sr);

            if (!stream) {
                // ---- non-streaming: wait for the full completion ----
                std::vector<int> gen;
                for (;;) {
                    int tok;
                    bool fin;
                    if (!engine.next_token(id, tok, fin, /*timeout_ms=*/5000))
                        continue;  // no data yet — keep waiting
                    if (fin) break;
                    gen.push_back(tok);
                }
                // Drop a trailing EOS token before decoding.
                std::vector<int> dec = gen;
                if (!dec.empty() && dec.back() == eos) dec.pop_back();
                std::string text = tokenizer.decode(dec);
                std::string body =
                    "{\"choices\":[{\"text\":\"" + json_escape(text) +
                    "\",\"finish_reason\":\"stop\"}]}";
                w.send(200, "application/json", body);
                return;
            }

            // ---- streaming: SSE over chunked transfer-encoding ----
            w.begin_stream(200, "text/event-stream");
            for (;;) {
                int tok;
                bool fin;
                if (!engine.next_token(id, tok, fin, /*timeout_ms=*/5000)) {
                    w.send_chunk("data: {\"choices\":[]}\n\n");  // keep-alive
                    continue;
                }
                if (fin) break;
                if (tok == eos) continue;  // don't stream the EOS token

                std::string chunk = tokenizer.decode({tok});
                std::string sse = "data: {\"choices\":[{\"delta\":{\"content\":\"" +
                                  json_escape(chunk) + "\"}}]}\n\n";
                w.send_chunk(sse);
            }
            w.send_chunk("data: [DONE]\n\n");
            w.end_stream();
            return;
        }

        w.send(404, "application/json", "{\"error\":\"not found\"}");
    });

    return 0;
}
