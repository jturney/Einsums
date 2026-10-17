//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/StandIns.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

namespace {
struct Record {
    void               *identity{nullptr};
    std::weak_ptr<void> caller_token;
    std::weak_ptr<void> owner;
};

struct Registry {
    std::mutex                               mutex;
    std::unordered_map<void const *, Record> stand_ins;
};

Registry &registry() {
    static Registry instance;
    return instance;
}
} // namespace

void note_stand_in(void const *stand_in, void *identity, std::weak_ptr<void> caller_token, std::weak_ptr<void> owner) {
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    reg.stand_ins.insert_or_assign(stand_in,
                                   Record{.identity = identity, .caller_token = std::move(caller_token), .owner = std::move(owner)});
}

void forget_stand_in(void const *stand_in) noexcept {
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    reg.stand_ins.erase(stand_in);
}

std::optional<StandIn> stand_in_of(void const *object) {
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    auto const             hit = reg.stand_ins.find(object);
    if (hit == reg.stand_ins.end()) {
        return std::nullopt;
    }
    // A stand-in whose last owner has let go is on its way out; its mark has not run yet.
    std::shared_ptr<void> owner = hit->second.owner.lock();
    if (owner == nullptr) {
        return std::nullopt;
    }
    return StandIn{.identity = hit->second.identity, .caller_token = hit->second.caller_token, .owner = std::move(owner)};
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
