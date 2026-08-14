/// End-to-end text generation with Qwen2.5 via EngineServer + BatchMainLoop.
/// Single user: tokenize prompt -> submit -> run -> decode generated tokens.
/// (Previously a hand-rolled forward; unified onto the EngineServer path.)
#include <cstdio>
#include <string>
#include <vector>

#include "lightllm/engine/engine.h"
#include "lightllm/engine/batch_loop.h"
#include "lightllm/tokenizer/tokenizer.h"

using namespace lightllm::engine;
using namespace lightllm::kv_cache;

int main(int argc, char** argv) {
    const char* model_dir = argc > 1 ? argv[1] : "models/qwen2.5-0.5b";
    bool use_fp16 = false;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--fp16")) use_fp16 = true;

    EngineServer engine(model_dir, /*max_seq_len=*/0, /*max_batch_tokens=*/256,
                        /*kv_cache_mb=*/0, prefix_cache_policy_from_env(), use_fp16);
    lightllm::tokenizer::Tokenizer tok(std::string(model_dir) + "/tokenizer.json");

    const char* text = "The capital of France is";
    std::vector<int> prompt = tok.encode(text);
    printf("Prompt: \"%s\" ->", text);
    for (int id : prompt) printf(" %d", id);
    printf("\n");

    BatchMainLoop loop(engine, SchedulerPolicy::FCFS, /*chunk_size=*/16, /*max_batch_tokens=*/256);
    int id = loop.submit(prompt, /*max_new=*/12, /*eos=*/151643, "", "", /*temperature=*/0.8f);
    loop.run();

    std::vector<int> gen = loop.generated_tokens(id);
    double ms = loop.metrics(id).latency_ms();
    printf("Generated %zu tokens in %.1f ms (%.1f tok/s)\n",
           gen.size(), ms, gen.size() * 1000.0 / (ms > 0 ? ms : 1.0));
    printf("Generated tokens:");
    for (int t : gen) printf(" %d", t);
    printf("\n");
    printf("Generated Text: %s\n", tok.decode(gen).c_str());
    return 0;
}
