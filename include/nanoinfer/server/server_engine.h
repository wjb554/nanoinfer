#pragma once
/// Thread-safe serving engine for NanoInfer.
///
/// Owns an EngineServer + Scheduler and runs the continuous batching loop on a
/// dedicated background thread (mirroring tools/bench_server.cpp).  HTTP-facing
/// threads interact with the engine ONLY through submit() / next_token() /
/// metrics(); the EngineServer and Scheduler are touched exclusively by the one
/// serving thread, which is what makes concurrent HTTP safe.
///
/// Threading model:
///   - engine_ and sched_ (and states_) are accessed ONLY by serving_thread_.
///   - pending_ queue guarded by pending_mu_.
///   - streams_ (id -> ReqStream) guarded by streams_mu_; each ReqStream's own
///     mutex + condition_variable guards token delivery to next_token().
///   - metrics_ counters are atomic; the TTFT/TPOT ring is guarded by metrics_.m.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nanoinfer {
namespace engine {
class EngineServer;
class Scheduler;
struct RequestState;   // NOTE: 'struct', not 'class' — MSVC mangles them
                       // differently, and engine.h declares RequestState as a
                       // struct; a 'class' forward decl here would break linking
}  // namespace engine

namespace server {

// ============================================================================
// ServeRequest — one generation request
// ============================================================================
struct ServeRequest {
    std::vector<int> prompt_tokens;
    int max_new_tokens = 32;
    int eos_token_id = 151643;
    float temperature = 1.0f;
};

// ============================================================================
// ServeMetrics — atomic counters + recent TTFT/TPOT ring buffers
// ============================================================================
struct ServeMetrics {
    std::atomic<long long> total_requests{0};
    std::atomic<long long> completed_requests{0};
    std::atomic<long long> active_requests{0};
    std::atomic<long long> total_tokens{0};
    std::atomic<long long> total_steps{0};
    std::atomic<long long> total_prefill_tokens{0};
    std::atomic<long long> total_decode_steps{0};

    // Recent TTFT/TPOT (ms): fixed-size ring buffer guarded by m.
    std::vector<double> ttft_ms;
    std::vector<double> tpot_ms;
    mutable std::mutex m;   // mutable so const readers can lock for the ring
    size_t ring_cap = 1000;

    /// Append a TTFT sample (ms) to the ring.
    void push_ttft(double v);
    /// Append a TPOT sample (ms) to the ring.
    void push_tpot(double v);
    /// Recent TTFT stats: average + max over the ring (0 / count 0 when empty).
    void recent_ttft(double& avg, double& maxv, int& count) const;
    /// Recent TPOT stats: average + max over the ring (0 / count 0 when empty).
    void recent_tpot(double& avg, double& maxv, int& count) const;
};

// ============================================================================
// ServerEngine
// ============================================================================
class ServerEngine {
public:
    /// @param max_batch_tokens per-step token budget (0 = VRAM-derived);
    ///                         applied to BOTH the engine's activation budget
    ///                         and the DecodeFirst scheduler.
    /// @param chunk_size       max prompt tokens per prefill entry
    ///                         (0 = VRAM-derived).
    ServerEngine(const std::string& model_dir, int max_seq_len,
                 int max_batch_tokens, bool use_fp16, int chunk_size = 0);
    ~ServerEngine();

    ServerEngine(const ServerEngine&) = delete;
    ServerEngine& operator=(const ServerEngine&) = delete;

    /// Spawn the serving thread (idempotent).
    void start();
    /// Stop and join the serving thread.
    void stop();

    /// Submit a request.  Thread-safe.  Returns a fresh request id (atomic
    /// counter).  Does NOT block on the engine.
    int submit(const ServeRequest& req);

    /// Streaming read: blocks up to timeout_ms for the next generated token of
    /// request_id.  Sets token_out (the token id), finished=true when the
    /// request is done (EOS / max_new_tokens).  Returns false only on timeout
    /// with no data.
    bool next_token(int request_id, int& token_out, bool& finished,
                    int timeout_ms);

    const ServeMetrics& metrics() const { return metrics_; }
    int vocab_size() const { return vocab_size_; }

private:
    void serving_loop();

    std::unique_ptr<engine::EngineServer> engine_;
    std::unique_ptr<engine::Scheduler> sched_;
    std::unordered_map<int, std::unique_ptr<engine::RequestState>> states_;

    /// Per-request delivery channel (created by submit(), read by next_token()
    /// and the serving thread).
    struct ReqStream {
        std::deque<int> tokens;
        std::mutex m;
        std::condition_variable cv;
        bool finished = false;
        std::vector<int> full_tokens;   // every generated token (incl. EOS)
        double ttft_ms = 0;             // set once at the first token
        std::chrono::steady_clock::time_point created_at;
        std::chrono::steady_clock::time_point last_token_at;
    };
    std::unordered_map<int, std::shared_ptr<ReqStream>> streams_;
    std::mutex streams_mu_;

    struct Pending { int id; ServeRequest req; };
    std::deque<Pending> pending_;
    std::mutex pending_mu_;

    std::thread serving_thread_;
    std::atomic<bool> running_{false};
    std::atomic<int> next_id_{0};
    int vocab_size_ = 0;

    ServeMetrics metrics_;
};

}  // namespace server
}  // namespace nanoinfer
