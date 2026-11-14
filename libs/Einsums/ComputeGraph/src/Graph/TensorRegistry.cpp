//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file TensorRegistry.cpp
/// @brief The tensor table's own invariants: ids, the by-address index, ownership, leases.

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <unordered_set>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {

TensorHandle const *TensorRegistry::find(TensorId id) const noexcept {
    auto const it = _tensors.find(id);
    return it == _tensors.end() ? nullptr : &it->second;
}

TensorHandle *TensorRegistry::find(TensorId id) noexcept {
    auto const it = _tensors.find(id);
    return it == _tensors.end() ? nullptr : &it->second;
}

TensorHandle const &TensorRegistry::insert(TensorHandle handle) {
    TensorId const id  = _next_tensor_id++;
    handle.id          = id;
    auto const &stored = _tensors.emplace(id, std::move(handle)).first->second;
    if (stored.tensor_ptr != nullptr) {
        // insert_or_assign, not emplace: an address freed during a capture can
        // be reused by a different tensor, and the index has to name the tensor
        // that lives there NOW. The stale-entry case is caught upstream by the
        // liveness-token check in CaptureContext::get_or_register, which is
        // what routes a recycled address here for re-registration.
        _ptr_index.insert_or_assign(stored.tensor_ptr, id);
    }
    // The caller's address only while the caller lives: once it is gone the address may already
    // belong to a tensor this graph indexed since, and that entry is the one that is right now.
    if (stored.caller_ptr != nullptr && stored.caller_ptr != stored.tensor_ptr && !stored.caller_token.expired()) {
        _ptr_index.insert_or_assign(stored.caller_ptr, id);
    }
    if (stored.owner != nullptr && stored.owner.get() != stored.tensor_ptr) {
        _ptr_index.insert_or_assign(stored.owner.get(), id);
    }
    return stored;
}

TensorId TensorRegistry::indexed(void const *ptr) const noexcept {
    auto const it = _ptr_index.find(ptr);
    return it == _ptr_index.end() ? 0 : it->second;
}

void TensorRegistry::index(void const *ptr, TensorId id) {
    _ptr_index.insert_or_assign(ptr, id);
}

void TensorRegistry::unindex(TensorId id, void const *ptr) {
    if (auto const stale = _ptr_index.find(ptr); stale != _ptr_index.end() && stale->second == id) {
        _ptr_index.erase(stale);
    }
}

void TensorRegistry::mark_host_resident(TensorId id) noexcept {
    if (TensorHandle *handle = find(id); handle != nullptr) {
        handle->residency = Residency::Host;
    }
}

bool TensorRegistry::add_scope_map(TensorScopeMapPtr map) {
    if (std::ranges::find(_scope_maps, map) != _scope_maps.end()) {
        return false;
    }
    _scope_maps.push_back(std::move(map));
    return true;
}

void TensorRegistry::own(void *ptr, void (*deleter)(void *)) {
    _owned_tensors.emplace_back(ptr, deleter);
    _owned_tensor_ptrs.insert(static_cast<void const *>(ptr));
}

void TensorRegistry::disown_last(void const *ptr) {
    _owned_tensor_ptrs.erase(ptr);
    _owned_tensors.pop_back();
}

TensorSlot *TensorRegistry::find_slot(TensorId id) noexcept {
    auto const it = _slot_map.find(id);
    return it != _slot_map.end() ? it->second.get() : nullptr;
}

void TensorRegistry::lease_scratch(std::string const &graph_name, std::shared_ptr<RunContext> const &context) {
    if (_scratch_leased_below == _next_tensor_id) {
        return;
    }
    for (auto &[tid, handle] : _tensors) {
        if (handle.scratch_leased || !handle.is_intermediate || !handle.storage_of) {
            continue;
        }
        handle.scratch_leased = true;
        auto *storage         = handle.storage_of();
        if (storage == nullptr || storage->scratch != nullptr) {
            continue;
        }
        storage->scratch = std::make_shared<einsums::detail::ScratchLease const>(
            einsums::detail::ScratchLease{.graph = graph_name, .state = context, .replaying = [](void const *state) noexcept {
                                              return static_cast<RunContext const *>(state)->replaying();
                                          }});
    }
    _scratch_leased_below = _next_tensor_id;
}

std::vector<TensorId> const &TensorRegistry::resync_ids(std::span<Node const> nodes, std::uint64_t version) {
    if (!_resync_ids_valid || _resync_ids_version != version || _resync_ids_nodes != nodes.size()) {
        std::unordered_set<TensorId> seen;
        _resync_ids.clear();
        for (Node const &node : nodes) {
            for (auto const *ids : {&node.inputs, &node.outputs}) {
                for (TensorId const id : *ids) {
                    if (seen.insert(id).second) {
                        _resync_ids.push_back(id);
                    }
                }
            }
        }
        _resync_ids_version = version;
        _resync_ids_nodes   = nodes.size();
        _resync_ids_valid   = true;
    }
    return _resync_ids;
}

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
