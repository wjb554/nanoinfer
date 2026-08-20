/// Tokenizer test — verify vocab load + decode + encode.
/// Runs for both tokenizer styles:
///   - GPT-2 byte-level (Qwen2): space is "Ġ" (U+0120)
///   - SentencePiece-style (Llama/Mistral): space is "▁" (U+2581)
/// The Llama path runs only when models/tinylama-1.1b-chat/ is present
/// (it is on the GPU server; local only has Qwen2.5).
#include <cstdio>
#include <cstring>
#include <string>
#include "nanoinfer/tokenizer/tokenizer.h"
using namespace nanoinfer::tokenizer;

int main(int argc, char** argv){
    printf("=== Tokenizer Test ===\n");
    std::string model_dir = argc > 1 ? argv[1] : "models/qwen2.5-0.5b";
    std::string tok_path = model_dir + "/tokenizer.json";

    Tokenizer tok(tok_path.c_str());
    printf("Vocab size: %d\n", tok.vocab_size());

    // Decode round-trip: encode then decode must restore the text.
    // (SP-style prepends ▁ so the result has a leading space; that is checked
    // separately in the SP branch below.)
    {
        std::string prompt = "The capital of France is";
        auto ids = tok.encode(prompt);
        std::string round = tok.decode(ids);
        printf("[encode] \"%s\" -> %d ids -> decode \"%s\"\n",
               prompt.c_str(), (int)ids.size(), round.c_str());
        bool sp = (std::strstr(tok_path.c_str(), "tinylama") != nullptr);
        if (sp) {
            if (round != " " + prompt) {
                printf("FAIL: round-trip mismatch (SP expects leading space)\n");
                return 1;
            }
        } else if (round != prompt) {
            printf("FAIL: round-trip mismatch\n");
            return 1;
        }
    }

    // SP-style (Llama): TinyLlama's vocab uses ▁.  HF encodes
    // "The capital of France is" as 1 450 7483 310 3444 338 (with BOS).
    // Our encode() does not emit BOS, so check the 5 content tokens match HF's
    // without the leading BOS (450 7483 310 3444 338).
    //
    // NOTE: the HF normalizer PREPENDS a "▁" to the text (Prepend normalizer),
    // so decode() of these tokens yields " The capital of France is" (leading
    // space) — that is correct SentencePiece behaviour, not a bug.
    if (std::strstr(tok_path.c_str(), "tinylama")) {
        std::string prompt = "The capital of France is";
        auto ids = tok.encode(prompt);
        printf("[sp-encode] ids:");
        for (int id : ids) printf(" %d", id);
        printf("\n");
        const int expected[] = {450, 7483, 310, 3444, 338};
        if ((int)ids.size() != 5) {
            printf("FAIL: expected 5 content tokens, got %d\n", (int)ids.size());
            return 1;
        }
        for (int i = 0; i < 5; i++) {
            if (ids[i] != expected[i]) {
                printf("FAIL: token[%d] = %d, expected %d\n", i, ids[i], expected[i]);
                return 1;
            }
        }
        // decode must restore "The capital..." with the prepend ▁ as a space
        std::string round = tok.decode(ids);
        printf("[sp-decode] \"%s\"\n", round.c_str());
        if (round != " The capital of France is") {
            printf("FAIL: sp-decode mismatch (expected leading space)\n");
            return 1;
        }
        printf("[sp] matches HF content tokens + decode\n");
    }

    printf("ALL PASSED\n");
    return 0;
}
