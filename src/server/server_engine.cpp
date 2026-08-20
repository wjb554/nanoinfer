/// Thread-safe serving engine — continuous batching on a background thread.
/// The loop below mirrors tools/bench_server.cpp: drain pending requests into
/// the scheduler, step the scheduler, execute the batch on the engine, and
/// fan sampled tokens out to per-request streams.

#include "nanoinfer/server/server_engine.h"

#include <chrono>

#include "nanoinfer/engine/engine.h"
#include "nanoinfer/engine/scheduler.h"
#include "nanoinfer/kv_cache/block_allocator.h"
#include "nanoinfer/kv_cache/prefix_cache.h"
#include "nanoinfer/model/model_config.h"

namespace nanoinfer {
namespace server {

using namespace nanoinfer::engine;
using Clock = std::chrono::steady_clock;

// ============================================================================
// ServeMetrics — ring-buffer helpers
// ============================================================================
static void push_ring(std::vector<double>& ring, size_t cap, double v) {
    if (ring.size() >= cap) ring.erase(ring.begin());
    ring.push_back(v);
}

void ServeMetrics::push_ttft(double v) {
    std::lock_guard<std::mutex> lk(m);
    push_ring(ttft_ms, ring_cap, v);
}

void ServeMetrics::push_tpot(double v) {
    std::lock_guard<std::mutex> lk(m);
    push_ring(tpot_ms, ring_cap, v);
}

static void recent_stats(const std::vector<double>& ring,
                         double& avg, double& maxv, int& count) {
    count = static_cast<int>(ring.size());
    avg = 0.0;
    maxv = 0.0;
    if (ring.empty()) return;
    double sum = 0.0;
    for (double v : ring) {
        sum += v;
        if (v > maxv) maxv = v;
    }
    avg = sum / static_cast<double>(ring.size());
}

void ServeMetrics::recent_ttft(double& avg, double& maxv, int& count) const {
    std::lock_guard<std::mutex> lk(m);
    recent_stats(ttft_ms, avg, maxv, count);
}

void ServeMetrics::recent_tpot(double& avg, double& maxv, int& count) const {
    std::lock_guard<std::mutex> lk(m);
    recent_stats(tpot_ms, avg, maxv, count);
}

// ============================================================================
// ServerEngine
// ============================================================================
ServerEngine::ServerEngine(const std::string& model_dir, int max_seq_len,
                           int max_batch_tokens, bool use_fp16, int chunk_size)
    : engine_(std::make_unique<EngineServer>(
          model_dir, max_seq_len, max_batch_tokens,
          /*kv_cache_mb=*/0, kv_cache::prefix_cache_policy_from_env(), use_fp16)),
      // Pass max_batch_tokens down to the scheduler so the server batch is not
      // silently capped at the factory's old hardcoded 256.  chunk_size 0 =
      // VRAM-derived default.
      sched_(Scheduler::create(SchedulerPolicy::DecodeFirst, chunk_size,
                               max_batch_tokens, /*max_prefill_entries=*/999)) {
    vocab_size_ = engine_->vocab_size();
}

ServerEngine::~ServerEngine() {
    stop();
    // Release any KV blocks still owned by unfinished requests (engine_ is
    // still valid until the member destructors run after this body).
    for (auto& [id, st] : states_) {
        engine_->release_request(*st);
    }
    states_.clear();
}

void ServerEngine::start() {
    if (running_.exchange(true)) return;  // already running
    serving_thread_ = std::thread([this] { serving_loop(); });
}

void ServerEngine::stop() {
    running_.store(false);
    if (serving_thread_.joinable()) serving_thread_.join();
}

int ServerEngine::submit(const ServeRequest& req) {
    int id = next_id_.fetch_add(1);

    auto stream = std::make_shared<ReqStream>();
    auto now = Clock::now();
    stream->created_at = now;
    stream->last_token_at = now;
    {
        std::lock_guard<std::mutex> lk(streams_mu_);
        streams_[id] = std::move(stream);
    }
    {
        std::lock_guard<std::mutex> lk(pending_mu_);
        pending_.push_back({id, req});
    }
    metrics_.total_requests++;
    metrics_.active_requests++;
    return id;
}

bool ServerEngine::next_token(int request_id, int& token_out, bool& finished,
                              int timeout_ms) {
    std::shared_ptr<ReqStream> stream;
    {
        std::lock_guard<std::mutex> lk(streams_mu_);
        auto it = streams_.find(request_id);
        if (it == streams_.end()) return false;  // unknown request id
        stream = it->second;
    }

    std::unique_lock<std::mutex> lk(stream->m);
    while (stream->tokens.empty() && !stream->finished) {
        if (stream->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms)) ==
            std::cv_status::timeout) {
            // Re-check under the lock: a token may have landed between the
            // spurious wake-up and the lock being re-acquired.
            if (stream->tokens.empty() && !stream->finished) return false;
            break;
        }
    }
    if (!stream->tokens.empty()) {
        token_out = stream->tokens.front();
        stream->tokens.pop_front();
        finished = false;
        return true;
    }
    // finished and no tokens remain
    finished = true;
    return true;
}

void ServerEngine::serving_loop() {
    while (running_.load()) {
        // 1. Drain pending requests into the scheduler.
        {
            std::lock_guard<std::mutex> lk(pending_mu_);
            while (!pending_.empty()) {
                Pending p = std::move(pending_.front());
                pending_.pop_front();

                Request req;
                req.id = p.id;
                req.prompt_tokens = std::move(p.req.prompt_tokens);
                req.max_new_tokens = p.req.max_new_tokens;
                req.eos_token_id = p.req.eos_token_id;
                req.temperature = p.req.temperature;

                states_[p.id] =
                    engine_->create_request_state(req, engine_->hidden_dim());
                sched_->add_request(std::move(req));
            }
        }

        // 2. Produce the next batch; sleep briefly when there is no work.
        ScheduleStep step = sched_->step();
        if (step.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 3. Execute the batch.  This is the ONLY place the engine is stepped.
        metrics_.total_steps++;
        for (auto& e : step.entries) {
            if (e.is_prefill) metrics_.total_prefill_tokens += e.num_tokens;
            else              metrics_.total_decode_steps++;
        }

        std::vector<SampledToken> sampled = engine_->step(step, states_);

        // 4. Fan tokens out to their streams.
        //
        //    EngineServer::step() pre-samples the FIRST output token during the
        //    final prefill chunk and stores it in RequestState::generated_tokens
        //    — it is NOT returned in `sampled` (decode entries start at the 2nd
        //    token).  Deliver that first token too, or the client's generation is
        //    missing its opening token (e.g. "The capital of France is" would
        //    answer ". It is the largest..." instead of "Paris. It is ...").
        //
        //    Build the full delivery list first: pre-sampled first tokens for
        //    requests whose prefill just completed (once — the stream starts
        //    empty), then this step's decode tokens.
        struct Deliv { int request_id; int token_id; bool is_eos; };
        std::vector<Deliv> to_deliver;
        {
            std::lock_guard<std::mutex> lk(streams_mu_);
            for (const auto& e : step.entries) {
                if (!e.is_prefill) continue;
                auto it = streams_.find(e.request_idx);
                if (it == streams_.end()) continue;  // unknown/dropped request
                auto st = states_.find(e.request_idx);
                if (st == states_.end()) continue;
                const auto& state = *st->second;
                // Prefill complete + a first token was pre-sampled + the stream
                // has not yet received it (full_tokens empty => not delivered).
                if (state.num_prefilled !=
                    static_cast<int>(state.prompt_tokens.size())) continue;
                if (state.generated_tokens.empty()) continue;
                if (!it->second->full_tokens.empty()) continue;
                // max_new_tokens reached by the pre-sampled first token alone
                // (e.g. max_new_tokens=1): treat it as the finishing token so
                // the request does not hang waiting for a decode EOS.
                bool first_only =
                    static_cast<int>(state.generated_tokens.size()) >=
                    state.max_new_tokens;
                to_deliver.push_back(
                    {e.request_idx, state.generated_tokens[0], first_only});
            }
        }
        for (auto& tok : sampled) {
            to_deliver.push_back({tok.request_id, tok.token_id, tok.is_eos});
        }

        for (auto& d : to_deliver) {
            metrics_.total_tokens++;

            std::shared_ptr<ReqStream> stream;
            {
                std::lock_guard<std::mutex> lk(streams_mu_);
                auto it = streams_.find(d.request_id);
                if (it == streams_.end()) continue;  // unknown/dropped request
                stream = it->second;
            }

            {
                std::lock_guard<std::mutex> lk(stream->m);
                stream->tokens.push_back(d.token_id);
                stream->full_tokens.push_back(d.token_id);

                auto now = Clock::now();
                if (stream->ttft_ms == 0.0) {
                    // First generated token -> TTFT (submit → first token).
                    stream->ttft_ms =
                        std::chrono::duration<double, std::milli>(
                            now - stream->created_at).count();
                    metrics_.push_ttft(stream->ttft_ms);
                } else {
                    // Subsequent token -> TPOT (inter-token latency).
                    double tpot =
                        std::chrono::duration<double, std::milli>(
                            now - stream->last_token_at).count();
                    metrics_.push_tpot(tpot);
                }
                stream->last_token_at = now;
            }
            stream->cv.notify_all();

            if (d.is_eos) {
                // 5. Free the request: engine state + scheduler bookkeeping.
                auto sit = states_.find(d.request_id);
                if (sit != states_.end()) {
                    engine_->release_request(*sit->second);
                    states_.erase(sit);
                }
                sched_->finish_request(d.request_id);

                metrics_.completed_requests++;
                metrics_.active_requests--;
                {
                    std::lock_guard<std::mutex> lk(stream->m);
                    stream->finished = true;
                }
                stream->cv.notify_all();
            }
        }
    }
}

}  // namespace server
}  // namespace nanoinfer
