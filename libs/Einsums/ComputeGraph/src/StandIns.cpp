//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/StandIns.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

namespace {
struct Record {
    void               *identity{nullptr};
    void               *caller{nullptr};
    std::weak_ptr<void> caller_token;
    std::weak_ptr<void> owner;
};

struct Registry {
    std::mutex                                     mutex;
    std::unordered_map<void const *, Record>       stand_ins;
    std::unordered_map<void const *, void const *> by_caller; ///< Caller's tensor to its stand-in.
    std::unordered_set<void const *>               graph_owned;
};

Registry &registry() {
    static Registry instance;
    return instance;
}
} // namespace

void note_stand_in(void const *stand_in, void *identity, void *caller, std::weak_ptr<void> caller_token, std::weak_ptr<void> owner) {
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    reg.stand_ins.insert_or_assign(
        stand_in, Record{.identity = identity, .caller = caller, .caller_token = std::move(caller_token), .owner = std::move(owner)});
    reg.by_caller.insert_or_assign(caller, stand_in);
}

void forget_stand_in(void const *stand_in) noexcept {
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    auto const             hit = reg.stand_ins.find(stand_in);
    if (hit == reg.stand_ins.end()) {
        return;
    }
    // Only while it still points here: the caller's address may already belong to a tensor with a
    // stand-in of its own.
    if (auto const back = reg.by_caller.find(hit->second.caller); back != reg.by_caller.end() && back->second == stand_in) {
        reg.by_caller.erase(back);
    }
    reg.stand_ins.erase(hit);
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
    return StandIn{.identity     = hit->second.identity,
                   .caller       = hit->second.caller,
                   .caller_token = hit->second.caller_token,
                   .owner        = std::move(owner)};
}

std::shared_ptr<void> stand_in_for(void const *caller, std::weak_ptr<void> const &caller_token) {
    if (untracked(caller_token)) {
        return {};
    }
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    auto const             back = reg.by_caller.find(caller);
    if (back == reg.by_caller.end()) {
        return {};
    }
    auto const hit = reg.stand_ins.find(back->second);
    if (hit == reg.stand_ins.end() || untracked(hit->second.caller_token) || !same_tensor(hit->second.caller_token, caller_token)) {
        return {};
    }
    return hit->second.owner.lock();
}

void note_graph_owned(void const *tensor) {
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    reg.graph_owned.insert(tensor);
}

void forget_graph_owned(void const *tensor) noexcept {
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    reg.graph_owned.erase(tensor);
}

bool is_graph_owned(void const *tensor) {
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    return reg.graph_owned.contains(tensor);
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
