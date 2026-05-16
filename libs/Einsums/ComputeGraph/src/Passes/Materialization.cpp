//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/DistributionDescriptor.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/Materialization.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace einsums::compute_graph::passes {

bool Materialization::run(Graph &graph) {
    _num_materialized = 0;
    _num_initialized  = 0;

    // Collect all deferred tensors.
    std::vector<TensorId> deferred_tids;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.alloc_state == AllocState::Deferred) {
            deferred_tids.push_back(tid);
        }
    }

    if (deferred_tids.empty())
        return false;

    auto &nodes = graph.nodes();

    // For each deferred tensor, find the earliest node that references it.
    std::unordered_map<TensorId, size_t> first_use;

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        for (auto tid : nodes[idx].inputs) {
            if (first_use.find(tid) == first_use.end())
                first_use[tid] = idx;
        }
        for (auto tid : nodes[idx].outputs) {
            if (first_use.find(tid) == first_use.end())
                first_use[tid] = idx;
        }
    }

    // Build insertion list: (position, nodes_to_insert).
    struct Insertion {
        size_t            position;
        std::vector<Node> new_nodes;
    };
    std::vector<Insertion> insertions;

    for (auto tid : deferred_tids) {
        auto &handle = graph.tensor(tid);

        // Determine insertion position: just before first use
        size_t insert_pos = 0;
        auto   use_it     = first_use.find(tid);
        if (use_it != first_use.end())
            insert_pos = use_it->second;

        std::vector<Node> new_nodes;

        // ── Materialize node ───────────────────────────────────────────────
        // Allocates storage when executed. Does NOT run during the pass.
        // For distributed tensors, resizes to local partition before allocating.
        {
            Node mat_node;
            mat_node.kind    = OpKind::Materialize;
            mat_node.label   = fmt::format("materialize({})", handle.name);
            mat_node.outputs = {tid};

            auto       mat_fn      = handle.materialize_fn;
            auto       resize_fn   = handle.resize_deferred_fn;
            auto       set_dist_fn = handle.set_distribution_fn;
            bool const is_dist     = handle.is_distributed && !handle.is_replicated;
            auto       dist_info   = handle.distribution_info;

            mat_node.execute = [mat_fn, resize_fn, set_dist_fn, is_dist, dist_info]() {
                if (is_dist && resize_fn && dist_info) {
                    auto      desc       = std::static_pointer_cast<comm::DistributionDescriptor>(dist_info);
                    int const rank       = comm::world_rank();
                    auto      local_dims = desc->local_dims_for(rank);
                    resize_fn(local_dims);

                    // Set distribution metadata on the tensor for T.range(dim) and T.global()
                    if (set_dist_fn) {
                        std::vector<size_t> offsets(desc->dim_to_axis.size());
                        for (size_t d = 0; d < desc->dim_to_axis.size(); d++) {
                            auto [start, end] = desc->local_range(d, rank);
                            offsets[d]        = start;
                        }
                        set_dist_fn(desc->global_dims, offsets);
                    }
                }
                if (mat_fn)
                    mat_fn();
            };

            new_nodes.push_back(std::move(mat_node));
            _num_materialized++;

            if (is_dist) {
                EINSUMS_LOG_INFO("Materialization: inserting materialize({}) [distributed] before position {}", handle.name, insert_pos);
            } else {
                EINSUMS_LOG_INFO("Materialization: inserting materialize({}) before position {}", handle.name, insert_pos);
            }
        }

        // ── Initialize node ────────────────────────────────────────────────
        // Fills tensor with zeros/random AFTER materialization.
        if (handle.init_kind != InitKind::None) {
            Node init_node;
            init_node.kind    = OpKind::Initialize;
            init_node.inputs  = {};    // No tensor inputs — runs right after Materialize
            init_node.outputs = {tid}; // Writes to the tensor (consumers depend on this)

            InitializeDescriptor desc;
            desc.tensor_id    = tid;
            desc.kind         = handle.init_kind;
            init_node.op_data = desc;

            if (handle.init_kind == InitKind::Zero) {
                init_node.label   = fmt::format("init_zero({})", handle.name);
                auto zero_fn      = handle.zero_fn;
                init_node.execute = [zero_fn]() {
                    if (zero_fn)
                        zero_fn();
                };
            } else if (handle.init_kind == InitKind::Random) {
                init_node.label   = fmt::format("init_random({})", handle.name);
                auto random_fn    = handle.random_fn;
                init_node.execute = [random_fn]() {
                    if (random_fn)
                        random_fn();
                };
            }

            new_nodes.push_back(std::move(init_node));
            _num_initialized++;
        }

        insertions.push_back({.position = insert_pos, .new_nodes = std::move(new_nodes)});
    }

    if (insertions.empty())
        return false;

    // Sort by position descending so inserts don't shift earlier positions.
    std::ranges::sort(insertions, [](Insertion const &a, Insertion const &b) { return a.position > b.position; });

    for (auto &ins : insertions) {
        nodes.insert(nodes.begin() + static_cast<ptrdiff_t>(ins.position), std::make_move_iterator(ins.new_nodes.begin()),
                     std::make_move_iterator(ins.new_nodes.end()));
    }

    graph.mark_sorted();

    return true;
}

} // namespace einsums::compute_graph::passes
