//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file HandleEdits.cpp
/// @brief The changes a pass may make to a registered tensor's handle, one method per change.
///
/// Each method owns the rules for its change and bumps the handle version when the handle
/// changes. They are the only writers of a registered handle outside the graph's own capture,
/// binding and annotation code.

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/TensorSlot.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// The handle registered under @p id, or a throw naming the graph and the id.
TensorHandle &registered(std::unordered_map<TensorId, TensorHandle> &tensors, std::string const &graph, TensorId id) {
    auto const it = tensors.find(id);
    if (it == tensors.end()) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Graph '{}': no tensor with id {}", graph, id);
    }
    return it->second;
}

} // namespace

bool Graph::infer_spaces(TensorId id, std::vector<SpaceId> spaces) {
    TensorHandle &handle = registered(_registry.handles(), _name, id);
    if (!handle.is_intermediate) {
        return false; // never change a tensor the caller owns
    }
    if (spaces.size() != handle.rank) {
        return false; // one space per axis, or nothing at all
    }
    if (!handle.spaces.empty()) {
        if (!handle.spaces_inferred) {
            return false; // a declaration is authoritative; inference never argues with it
        }
        if (handle.spaces == spaces) {
            return false; // already says this, so a re-run reports no new inference
        }
    }
    handle.spaces          = std::move(spaces);
    handle.spaces_inferred = true;
    note(Change::HandleChanged);
    return true;
}

bool Graph::infer_symmetry(TensorId id, SymmetryDescriptor descriptor, SymmetryReach reach) {
    TensorHandle &handle = registered(_registry.handles(), _name, id);
    if (reach == SymmetryReach::Tensor && !handle.is_intermediate) {
        return false; // never give the caller's own tensor a declaration it did not make
    }
    if (handle.symmetry_hint != nullptr && *handle.symmetry_hint == descriptor) {
        return false; // already says this, so a re-run reports no new inference
    }
    handle.symmetry_hint = std::make_shared<SymmetryDescriptor const>(descriptor);
    if (reach == SymmetryReach::Tensor && handle.ops->set_symmetry != nullptr) {
        handle.ops->set_symmetry(handle.live_ptr(), std::move(descriptor));
    }
    note(Change::HandleChanged);
    return true;
}

void Graph::relayout(TensorId id, std::span<std::size_t const> order) {
    TensorHandle &handle = registered(_registry.handles(), _name, id);
    if (handle.alloc_state != AllocState::Deferred || !handle.resizable || handle.ops->resize_deferred == nullptr) {
        EINSUMS_THROW_EXCEPTION(std::logic_error,
                                "Graph '{}': relayout of tensor '{}' needs a deferred tensor with a resize hook; re-laying out a live "
                                "buffer is a data movement, not a change of declaration",
                                _name, handle.name);
    }
    std::vector<bool> seen(handle.rank, false);
    bool              permutation = order.size() == handle.rank;
    for (std::size_t const axis : order) {
        permutation = permutation && axis < handle.rank && !seen[axis];
        if (axis < handle.rank) {
            seen[axis] = true;
        }
    }
    if (!permutation) {
        EINSUMS_THROW_EXCEPTION(std::logic_error,
                                "Graph '{}': relayout of tensor '{}' (rank {}) given ({}), which is not a permutation of its axes", _name,
                                handle.name, handle.rank, fmt::join(order, ","));
    }

    // Every per-axis annotation moves with the axis it describes. A missed one is not a wrong
    // number today; it is a dim symbol naming the wrong extent at the next bind, which is the
    // failure the whole symbolic-extent machinery exists to catch.
    auto const permuted = [order](auto const &per_axis) {
        std::remove_cvref_t<decltype(per_axis)> moved;
        moved.reserve(order.size());
        for (std::size_t const axis : order) {
            moved.push_back(per_axis[axis]);
        }
        return moved;
    };
    std::vector<std::size_t> const dims = permuted(handle.dims);
    if (handle.spaces.size() == order.size()) {
        handle.spaces = permuted(handle.spaces);
    }
    if (handle.dim_symbols.size() == order.size()) {
        handle.dim_symbols = permuted(handle.dim_symbols);
    }

    handle.ops->resize_deferred(handle.live_ptr(), dims);
    handle.dims = dims;
    handle.strides.assign(dims.size(), 1);
    for (std::size_t axis = 1; axis < dims.size(); axis++) {
        handle.strides[axis] = handle.strides[axis - 1] * dims[axis - 1]; // column-major
    }
    if (TensorSlot *slot = find_slot(id); slot != nullptr) {
        slot->dims = dims;
    }
    note(Change::HandleChanged);
}

void Graph::set_distribution(TensorId id, std::optional<comm::DistributionDescriptor> distributed) {
    TensorHandle &handle = registered(_registry.handles(), _name, id);
    if (distributed.has_value()) {
        handle.is_distributed    = true;
        handle.is_replicated     = false;
        handle.distribution_info = std::make_shared<comm::DistributionDescriptor>(std::move(*distributed));
    } else {
        // Cleared too: a reader that asks only whether there is a descriptor (SUMMAExpansion does)
        // must not find the last plan's on a tensor this plan replicates.
        handle.is_distributed = false;
        handle.is_replicated  = true;
        handle.distribution_info.reset();
    }
    note(Change::HandleChanged);
}

void Graph::adopt_lifecycle(TensorId id, TensorHandle const &from) {
    TensorHandle &handle  = registered(_registry.handles(), _name, id);
    handle.alloc_state    = from.alloc_state;
    handle.init_kind      = from.init_kind;
    handle.materialize_fn = from.materialize_fn;
    handle.zero_fn        = from.zero_fn;
    handle.random_fn      = from.random_fn;
    handle.custom_init    = from.custom_init;
    note(Change::HandleChanged);
}

void Graph::set_residency(TensorId id, Residency residency) {
    TensorHandle &handle = registered(_registry.handles(), _name, id);
    if (handle.residency != residency) {
        handle.residency = residency;
        note(Change::HandleChanged);
    }
}

void Graph::mark_materialized(TensorId id) {
    TensorHandle &handle = registered(_registry.handles(), _name, id);
    handle.alloc_state   = AllocState::Materialized;
    handle.init_kind     = InitKind::None;
    note(Change::HandleChanged);
}

EINSUMS_NAMESPACE_END(compute_graph)
