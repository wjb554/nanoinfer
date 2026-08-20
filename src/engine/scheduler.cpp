/// Scheduler implementations — FCFS, PrefillFirst.
#include "nanoinfer/engine/scheduler.h"
#include "nanoinfer/engine/tuning.h"
#include <algorithm>
#include <stdexcept>

namespace nanoinfer {
namespace engine {

// ===== Factory ==============================================================
std::unique_ptr<Scheduler> Scheduler::create(SchedulerPolicy policy,
                                             int chunk_size,
                                             int max_batch_tokens,
                                             int max_prefill_entries) {
    // 0 in either budget arg means "derive from VRAM" (engine/tuning.h), so a
    // caller like ServerEngine that just passes max_batch_tokens gets the same
    // derivation as the benchmark tools instead of a hardcoded 256.
    int chunk  = chunk_size       > 0 ? chunk_size       : default_chunk_size();
    int budget = max_batch_tokens > 0 ? max_batch_tokens : default_batch_tokens();
    switch (policy) {
        case SchedulerPolicy::FCFS:
            return std::make_unique<FCFSScheduler>(chunk);
        case SchedulerPolicy::PrefillFirst:
            return std::make_unique<PrefillFirstScheduler>(chunk);
        case SchedulerPolicy::DecodeFirst:
            return std::make_unique<DecodeFirstScheduler>(chunk, budget,
                                                          max_prefill_entries);
        default:
            throw std::runtime_error("Unknown scheduler policy");
    }
}

// ===== DecodeFirst Scheduler =================================================
DecodeFirstScheduler::DecodeFirstScheduler(int chunk_size, int max_batch_tokens,
                                         int max_prefill_entries)
    : chunk_size_(chunk_size), max_batch_tokens_(max_batch_tokens),
      max_prefill_entries_(max_prefill_entries) {}

void DecodeFirstScheduler::add_request(Request req) {
    req.state = Request::WAITING;
    waiting_.push_back(std::move(req));
}

ScheduleStep DecodeFirstScheduler::step() {
    ScheduleStep step;
    int token_budget = max_batch_tokens_;

    // Priority 1: All decode tokens (ensure interactive users don't stall)
    for (auto& req : decoding_) {
        if (req.state == Request::FINISHED) continue;
        step.entries.push_back({req.id, 1, req.total_len(), false});
        token_budget -= 1;
    }

    // Priority 2: Use remaining budget for prefill (capped by max entries)
    int prefill_count = 0;
    while (!waiting_.empty() && token_budget > 0
           && prefill_count < max_prefill_entries_) {
        auto& req = waiting_.front();
        int remaining = req.prefill_remaining();
        if (remaining <= 0) { waiting_.pop_front(); continue; }

        int chunk = std::min({chunk_size_, remaining, token_budget});
        step.entries.push_back({req.id, chunk, req.num_computed, true});
        req.num_computed += chunk;
        token_budget -= chunk;
        prefill_count++;

        if (req.prefill_remaining() <= 0) {
            req.state = Request::DECODING;
            decoding_.push_back(std::move(req));
            waiting_.pop_front();
        }
    }

    // Purge finished requests from decoding
    decoding_.erase(
        std::remove_if(decoding_.begin(), decoding_.end(),
            [](const Request& r){ return r.state == Request::FINISHED; }),
        decoding_.end());

    return step;
}

void DecodeFirstScheduler::finish_request(int req_id) {
    for (auto& req : decoding_)
        if (req.id == req_id) { req.state = Request::FINISHED; return; }
}

// ===== FCFS Scheduler =======================================================
FCFSScheduler::FCFSScheduler(int chunk_size) : chunk_size_(chunk_size) {}

void FCFSScheduler::add_request(Request req) {
    req.state = Request::WAITING;
    requests_.push_back(std::move(req));
}

ScheduleStep FCFSScheduler::step() {
    ScheduleStep step;

    // If current request is finished or null, pick next from queue
    if (!current_ || current_->state == Request::FINISHED) {
        if (requests_.empty()) return step;
        current_ = &requests_.front();
        current_->state = Request::WAITING;
    }

    auto& req = *current_;

    // Phase 1: Prefill remaining prompt tokens
    if (req.prefill_remaining() > 0) {
        int chunk = std::min(chunk_size_, req.prefill_remaining());
        step.entries.push_back({req.id, chunk, req.num_computed, true});
        req.num_computed += chunk;
        return step;
    }

    // Phase 2: All prefill done → decode one token
    if (req.state != Request::FINISHED) {
        req.state = Request::DECODING;
        step.entries.push_back({req.id, 1, req.total_len(), false});
        return step;
    }

    return step;
}

void FCFSScheduler::finish_request(int req_id) {
    // FCFS only processes one request at a time
    if (current_ && current_->id == req_id) {
        current_->state = Request::FINISHED;
        if (!requests_.empty()) requests_.pop_front();
        current_ = nullptr;
    }
}

// ===== PrefillFirst Scheduler ===============================================
PrefillFirstScheduler::PrefillFirstScheduler(int chunk_size) : chunk_size_(chunk_size) {}

void PrefillFirstScheduler::add_request(Request req) {
    req.state = Request::WAITING;
    waiting_.push_back(std::move(req));
}

bool PrefillFirstScheduler::prefill_one_chunk(ScheduleStep& step, Request& req) {
    int remaining = req.prefill_remaining();
    if (remaining <= 0) return false;
    int chunk = std::min(chunk_size_, remaining);
    step.entries.push_back({req.id, chunk, req.num_computed, true});
    req.num_computed += chunk;
    // If prefill done, move to decoding queue
    if (req.prefill_remaining() <= 0) {
        req.state = Request::DECODING;
        decoding_.push_back(std::move(req));
        return true; // moved
    }
    return false; // still in waiting, more chunks needed
}

void PrefillFirstScheduler::collect_decode_tokens(ScheduleStep& step) {
    for (auto& req : decoding_) {
        if (req.state == Request::FINISHED) continue;
        step.entries.push_back({req.id, 1, req.total_len(), false});
    }
}

ScheduleStep PrefillFirstScheduler::step() {
    ScheduleStep step;

    // Purge finished requests from decoding queue
    decoding_.erase(
        std::remove_if(decoding_.begin(), decoding_.end(),
            [](const Request& r){ return r.state == Request::FINISHED; }),
        decoding_.end());

    // Priority 1: Prefill one chunk from the oldest waiting request
    if (!waiting_.empty()) {
        auto& req = waiting_.front();
        bool moved = prefill_one_chunk(step, req);
        if (moved) waiting_.pop_front();
    }

    // Priority 2: Collect decode tokens from all decoding requests
    collect_decode_tokens(step);

    // If nothing to do and waiting is empty, we're idle
    return step;
}

void PrefillFirstScheduler::finish_request(int req_id) {
    for (auto& req : decoding_) {
        if (req.id == req_id) {
            req.state = Request::FINISHED;
            return;
        }
    }
}

}  // namespace engine
}  // namespace nanoinfer
