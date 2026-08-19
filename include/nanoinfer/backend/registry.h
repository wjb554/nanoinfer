#pragma once
/// Generic backend factory registry + registration helper macro.
///
/// Plugin architecture (P0/P1): each backend interface (IGemmBackend,
/// IAttentionBackend) owns a BackendRegistry<T> singleton.  Concrete
/// implementations register themselves via NANOINFER_REGISTER_BACKEND (one
/// anonymous-namespace registrar object per registration).  Selection is
/// env-driven in EngineServer's constructor (NANOINFER_GEMM_BACKEND /
/// NANOINFER_ATTN_BACKEND).  The default backends are the engine's original
/// implementations moved verbatim, so default execution is numerically
/// identical to before this refactor.

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nanoinfer {
namespace backend {

/// Thread-safe-per-TU generic factory registry for interface T.
template <class T>
class BackendRegistry {
public:
    using Factory = std::function<std::unique_ptr<T>()>;

    static BackendRegistry& instance() {
        static BackendRegistry inst;
        return inst;
    }

    /// Register a factory under `name`.  Returns false (and leaves the prior
    /// factory untouched) if the name is already registered.
    bool register_backend(const std::string& name, Factory f) {
        if (name.empty() || !f) return false;
        auto result = reg_.emplace(name, std::move(f));
        return result.second;
    }

    /// Create a backend by name.  Throws std::runtime_error if `name` is not
    /// a registered backend.
    std::unique_ptr<T> create(const std::string& name) const {
        auto it = reg_.find(name);
        if (it == reg_.end())
            throw std::runtime_error(
                "backend registry: no backend named '" + name + "'");
        return it->second();
    }

    /// All registered backend names (sorted, std::map order).
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(reg_.size());
        for (const auto& kv : reg_) out.push_back(kv.first);
        return out;
    }

private:
    std::map<std::string, Factory> reg_;
};

// ---------------------------------------------------------------------------
// Force-link plumbing.
//
// nanoinfer is a STATIC library.  The registrar objects produced by
// NANOINFER_REGISTER_BACKEND live in the backend .cpp TUs and register
// themselves at static-init time.  A static-archive linker only pulls an
// object file out of the archive when it resolves an undefined symbol, so the
// registrar TUs would be dropped unless something in the linked program
// references them.  EngineServer's constructor calls
// force_link_backend_registrars(), which references one non-inline symbol per
// backend TU, forcing the archive to include those objects (and their
// registrar side effects).
// ---------------------------------------------------------------------------
void force_link_gemm_backend();
void force_link_attention_backend();

inline void force_link_backend_registrars() {
    force_link_gemm_backend();
    force_link_attention_backend();
}

}  // namespace backend
}  // namespace nanoinfer

// Two-level token paste so Name / Impl (macro args) expand before
// concatenating into the registrar's identifier.
#define NANOINFER_BACKEND_CONCAT_INNER(a, b, c) a##b##c
#define NANOINFER_BACKEND_CONCAT(a, b, c) NANOINFER_BACKEND_CONCAT_INNER(a, b, c)

/// Registers `Impl` as an implementation of interface `Iface` under the
/// (stringified) name `Name`.  Uses an anonymous-namespace registrar object so
/// the registration runs at static-init time in any binary that links the TU.
/// `Name` must be a valid identifier (it participates in the registrar's
/// symbol name); `Impl` must name a concrete type constructible with no args.
#define NANOINFER_REGISTER_BACKEND(Iface, Name, Impl)                            \
    namespace {                                                                 \
    struct NANOINFER_BACKEND_CONCAT(nanoinfer_reg_, Name, Impl) {               \
        NANOINFER_BACKEND_CONCAT(nanoinfer_reg_, Name, Impl)() {                \
            nanoinfer::backend::BackendRegistry<Iface>::instance()             \
                .register_backend(#Name,                                        \
                    [] { return std::make_unique<Impl>(); });                   \
        }                                                                       \
    };                                                                          \
    NANOINFER_BACKEND_CONCAT(nanoinfer_reg_, Name, Impl)                        \
        NANOINFER_BACKEND_CONCAT(nanoinfer_reg_inst_, Name, Impl);              \
    }  // namespace
