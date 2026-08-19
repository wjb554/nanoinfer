/// NanoInfer HTTP Inference Server — loads model, serves requests.
/// POST /v1/chat/completions  {"prompt":[ids],"max_tokens":N} → {"tokens":[...]}
/// Backed by EngineServer + BatchMainLoop (single-threaded handler).
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include "nanoinfer/engine/engine.h"
#include "nanoinfer/engine/batch_loop.h"
#include "nanoinfer/server/http_server.h"

using namespace nanoinfer::engine;
using namespace nanoinfer::server;

// Minimal JSON builder
static std::string json_obj(const std::string& content){
    return "{" + content + "}";
}
static std::string json_key(const std::string& k, const std::string& v){
    return "\"" + k + "\": " + v;
}
static std::string json_array(const std::vector<int>& ids){
    std::string s="[";
    for(size_t i=0;i<ids.size();i++){s+=std::to_string(ids[i]);if(i+1<ids.size())s+=",";}
    return s+"]";
}
// Minimal JSON parser — extracts value for key
static int json_int(const std::string& body, const std::string& key){
    auto p=body.find("\""+key+"\"");
    if(p==std::string::npos)return 0;
    p=body.find(':',p);
    while(p<body.size()&&(body[p]<'0'||body[p]>'9'))p++;
    return std::atoi(body.c_str()+p);
}
static std::vector<int> json_int_array(const std::string& body,const std::string& key){
    std::vector<int> v;
    auto p=body.find("\""+key+"\"");
    if(p==std::string::npos)return v;
    p=body.find('[',p);if(p==std::string::npos)return v;
    size_t e=body.find(']',p);
    for(size_t i=p+1;i<e;){while(i<e&&(body[i]<'0'||body[i]>'9'))i++;if(i>=e)break;size_t j=i;while(j<e&&body[j]>='0'&&body[j]<='9')j++;v.push_back(std::atoi(body.c_str()+i));i=j+1;}
    return v;
}

int main(){
    printf("=== NanoInfer HTTP Server ===\nLoading model...\n");
    EngineServer engine("models/qwen2.5-0.5b", /*max_seq_len=*/0, /*max_batch_tokens=*/256,
                        /*kv_cache_mb=*/0, nanoinfer::kv_cache::prefix_cache_policy_from_env(),
                        /*use_fp16=*/false);
    BatchMainLoop loop(engine, SchedulerPolicy::FCFS, /*chunk_size=*/16, /*max_batch_tokens=*/256);

    run_server(8080, [&](const HttpRequest& req, HttpResponseWriter& w) {
        if(req.path=="/health"){ w.send(200,"application/json","{\"status\":\"ok\"}"); return; }

        auto prompt=json_int_array(req.body,"prompt");
        int max_tok=json_int(req.body,"max_tokens");
        if(max_tok==0)max_tok=16;
        if(prompt.empty())prompt={1}; // BOS token

        printf("Request: %zu prompt tokens, max_tokens=%d\n",prompt.size(),max_tok);

        auto t0=std::chrono::steady_clock::now();
        int id = loop.submit(prompt, max_tok, /*eos*/151643, "", "", /*temperature*/1.0f);
        loop.run();
        auto gen = loop.generated_tokens(id);
        auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count();

        int new_tokens=(int)gen.size();
        printf("  Generated %d tokens in %lld ms\n",new_tokens,ms);

        // Full sequence = prompt + generated (matches the old generate() contract).
        std::vector<int> all = prompt;
        all.insert(all.end(), gen.begin(), gen.end());

        auto output=json_array(all);
        auto body=json_obj(
            json_key("tokens",output)+","+
            json_key("new_tokens",std::to_string(new_tokens))+","+
            json_key("time_ms",std::to_string(ms))
        );
        w.send(200, "application/json", body);
    });
    return 0;
}
