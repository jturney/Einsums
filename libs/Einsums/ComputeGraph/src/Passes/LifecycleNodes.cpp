//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "LifecycleNodes.hpp"

#include <Einsums/Comm/DistributionDescriptor.hpp>
#include <Einsums/Comm/Runtime.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <memory>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::lifecycle)

namespace {

AllocDescriptor describe(TensorHandle const &handle, TensorId emit_tid) {
    return AllocDescriptor{.tensor_id = emit_tid, .size_bytes = handle.total_bytes(), .tensor_name = handle.name};
}

} // namespace

Node make_materialize_node(TensorHandle const &handle, TensorId emit_tid) {
    Node node;
    node.kind            = OpKind::Materialize;
    node.label           = fmt::format("materialize({})", handle.name);
    node.outputs         = {emit_tid}; // WAW edge orders it before the first writer
    node.op_data         = describe(handle, emit_tid);
    node.estimated_bytes = handle.total_bytes();

    auto       mat_fn      = handle.materialize_fn;
    auto       resize_fn   = handle.resize_deferred_fn;
    auto       set_dist_fn = handle.set_distribution_fn;
    bool const is_dist     = handle.is_distributed && !handle.is_replicated;
    auto       dist_info   = handle.distribution_info;

    // Idempotent: a no-op when the tensor is already allocated, a reallocation on a replay after a
    // Free reclaimed it.
    node.execute = [mat_fn, resize_fn, set_dist_fn, is_dist, dist_info]() {
        if (is_dist && resize_fn && dist_info) {
            auto      desc       = std::static_pointer_cast<comm::DistributionDescriptor>(dist_info);
            int const rank       = comm::world_rank();
            auto      local_dims = desc->local_dims_for(rank);
            resize_fn(local_dims);

            if (set_dist_fn) {
                std::vector<size_t> offsets(desc->dim_to_axis.size());
                for (size_t d = 0; d < desc->dim_to_axis.size(); d++) {
                    auto [start, end] = desc->local_range(d, rank);
                    offsets[d]        = start;
                }
                set_dist_fn(desc->global_dims, offsets);
            }
        }
        if (mat_fn) {
            mat_fn();
        }
    };
    return node;
}

Node make_free_node(TensorHandle const &handle, TensorId emit_tid) {
    Node node;
    node.kind            = OpKind::Free;
    node.label           = fmt::format("free({})", handle.name);
    node.inputs          = {emit_tid};
    node.outputs         = {emit_tid};
    node.op_data         = describe(handle, emit_tid);
    node.estimated_bytes = handle.total_bytes();
    // No logging here: this fires once per freed tensor per REPLAY.
    node.execute = [rel_fn = handle.release_fn]() {
        if (rel_fn) {
            rel_fn();
        }
    };
    return node;
}

std::string_view lifecycle_tensor_name(Node const &node) noexcept {
    if (node.kind != OpKind::Alloc && node.kind != OpKind::Materialize && node.kind != OpKind::Free) {
        return {};
    }
    auto const *desc = node.op_data.get_if<AllocDescriptor>();
    return desc != nullptr ? std::string_view{desc->tensor_name} : std::string_view{};
}

bool has_lifecycle_node(std::span<Node const> nodes, OpKind kind, std::string_view name) noexcept {
    return std::ranges::any_of(nodes, [kind, name](Node const &node) { return node.kind == kind && lifecycle_tensor_name(node) == name; });
}

EINSUMS_NAMESPACE_END(compute_graph::passes::lifecycle)
