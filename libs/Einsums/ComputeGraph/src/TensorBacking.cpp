//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/InterfaceManifest.hpp>
#include <Einsums/ComputeGraph/TensorBacking.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <fmt/format.h>

#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {
struct Registry {
    std::shared_mutex                               mutex;
    std::map<std::string, BackingKind, std::less<>> kinds;
};

Registry &registry() {
    static Registry instance;
    return instance;
}
} // namespace

std::optional<std::string> TensorBacking::dtype_name() const {
    if (dtype == packed_gemm::ScalarType::Unknown) {
        return std::nullopt;
    }
    return std::string{scalar_type_name(dtype)};
}

std::string TensorBacking::describe() const {
    std::string where;
    for (auto const &[key, value] : location) {
        where += fmt::format("{}{}={}", where.empty() ? "" : ", ", key, value);
    }
    return fmt::format("{} ({})", kind, where);
}

void register_backing_kind(BackingKind kind) {
    if (kind.name.find('.') == std::string::npos) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "register_backing_kind: '{}' is not qualified; a backing kind's name must contain a '.', as in "
                                "'module.kind'",
                                kind.name);
    }
    if (!kind.open) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "register_backing_kind: '{}' needs an open function", kind.name);
    }
    auto                               &reg = registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    auto [it, inserted] = reg.kinds.try_emplace(kind.name, std::move(kind));
    if (!inserted) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "register_backing_kind: '{}' is already registered", it->first);
    }
}

TensorBacking open_backing(std::string const &kind, std::vector<std::pair<std::string, std::string>> const &location, bool write) {
    std::function<TensorBacking(std::vector<std::pair<std::string, std::string>> const &, bool)> open;
    {
        auto                               &reg = registry();
        std::shared_lock<std::shared_mutex> lock(reg.mutex);
        auto const                          it = reg.kinds.find(kind);
        if (it == reg.kinds.end()) {
            EINSUMS_THROW_EXCEPTION(
                std::invalid_argument,
                "no backing kind is registered under '{}'; the module that provides it has to be linked and initialized", kind);
        }
        open = it->second.open;
    }
    // Outside the lock: opening reads storage, and a kind that registers another from inside its
    // open would otherwise deadlock.
    TensorBacking backing = open(location, write);
    if (backing.kind != kind) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "backing kind '{}' opened a backing that names itself '{}'", kind, backing.kind);
    }
    if (backing.write != write) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "backing kind '{}' opened a {} backing for a {} location", kind,
                                backing.write ? "write" : "read", write ? "write" : "read");
    }
    if (write ? !backing.prepare : !backing.read_slab) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "backing kind '{}' opened a {} backing with no {}", kind, write ? "write" : "read",
                                write ? "prepare step" : "reader");
    }
    return backing;
}

EINSUMS_NAMESPACE_END(compute_graph)
