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

namespace {

// Build the Materialize (and optional Initialize) node for a single
// deferred tensor. The returned vector either has size 1 (Materialize
// only) or 2 (Materialize + Initialize).
//
// @param handle         Mutable handle whose ``materialize_fn`` /
//                       ``init_kind`` / distribution metadata drives the
//                       node's execute closure.
// @param owns_tid       True when the tensor's id lives in the *target*
//                       graph's tensor_map. When false (the hoisted
//                       case), we leave the synthetic node's outputs
//                       empty so the target graph's dep tracking doesn't
//                       try to resolve a foreign TensorId. Execution
//                       order is established by node position alone in
//                       that case.
std::vector<Node> build_lifecycle_nodes(TensorHandle &handle, bool owns_tid) {
    std::vector<Node> out;

    // ── Materialize ────────────────────────────────────────────────────
    {
        Node mat_node;
        mat_node.kind  = OpKind::Materialize;
        mat_node.label = fmt::format("materialize({})", handle.name);
        if (owns_tid) {
            mat_node.outputs = {handle.id};
        }

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
        out.push_back(std::move(mat_node));
    }

    // ── Initialize (optional) ──────────────────────────────────────────
    if (handle.init_kind != InitKind::None) {
        Node init_node;
        init_node.kind = OpKind::Initialize;
        if (owns_tid) {
            init_node.outputs = {handle.id};
        }

        InitializeDescriptor desc;
        desc.tensor_id    = handle.id;
        desc.kind         = handle.init_kind;
        init_node.op_data = desc;

        if (handle.init_kind == InitKind::Zero) {
            init_node.label   = fmt::format("init_zero({})", handle.name);
            auto zero_fn      = handle.zero_fn;
            init_node.execute = [zero_fn]() {
                if (zero_fn) {
                    zero_fn();
                }
            };
        } else if (handle.init_kind == InitKind::Random) {
            init_node.label   = fmt::format("init_random({})", handle.name);
            auto random_fn    = handle.random_fn;
            init_node.execute = [random_fn]() {
                if (random_fn) {
                    random_fn();
                }
            };
        }

        out.push_back(std::move(init_node));
    }

    return out;
}

// Walks descendants of @p graph (loop bodies, conditional branches) and
// every sub-graph nested inside them, in post-order. For each deferred
// tensor it finds, appends the (graph-that-owns-handle, tensor-id) pair
// to @p out so the caller can build hoisted lifecycle nodes for it. Does
// NOT collect @p graph's own deferred tensors — that's the caller's job.
struct DeferredEntry {
    Graph   *handle_owner;
    TensorId tid;
};

void collect_descendant_deferred(Graph &graph, std::vector<DeferredEntry> &out) {
    graph.for_each_subgraph([&](Graph &sub) {
        for (auto const &[tid, handle] : sub.tensors_map()) {
            if (handle.alloc_state == AllocState::Deferred) {
                out.push_back({.handle_owner = &sub, .tid = tid});
            }
        }
        collect_descendant_deferred(sub, out);
    });
}

} // namespace

bool Materialization::run(Graph &graph) {
    _num_materialized = 0;
    _num_initialized  = 0;

    // ── 1. The parent graph's own deferred tensors ────────────────────────
    std::vector<TensorId> own_deferred;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.alloc_state == AllocState::Deferred) {
            own_deferred.push_back(tid);
        }
    }

    // ── 2. Descendants' deferred tensors (loop bodies, conditional
    //      branches, and any nesting underneath). Each such tensor will
    //      be hoisted to the outermost parent so its lifecycle runs once
    //      per outer execution instead of every iteration / every
    //      branch entry.
    //
    //      We walk parent.nodes() directly (not for_each_subgraph) so we
    //      know which node index owns each sub-graph — that's where the
    //      hoisted Materialize / Initialize node goes.
    struct Hoist {
        size_t   owning_node_index;
        Graph   *handle_owner;
        TensorId tid;
    };
    std::vector<Hoist> hoists;

    auto const &parent_nodes = graph.nodes();
    for (size_t i = 0; i < parent_nodes.size(); i++) {
        Node const &node = parent_nodes[i];

        auto collect_from = [&](Graph &child) {
            for (auto const &[tid, handle] : child.tensors_map()) {
                if (handle.alloc_state == AllocState::Deferred) {
                    hoists.push_back({.owning_node_index = i, .handle_owner = &child, .tid = tid});
                }
            }
            std::vector<DeferredEntry> nested;
            collect_descendant_deferred(child, nested);
            for (auto const &e : nested) {
                hoists.push_back({.owning_node_index = i, .handle_owner = e.handle_owner, .tid = e.tid});
            }
        };

        if (auto const *loop = std::get_if<LoopDescriptor>(&node.op_data)) {
            if (loop->body) {
                collect_from(*loop->body);
            }
        } else if (auto const *cond = std::get_if<ConditionalDescriptor>(&node.op_data)) {
            if (cond->then_branch) {
                collect_from(*cond->then_branch);
            }
            if (cond->else_branch) {
                collect_from(*cond->else_branch);
            }
        }
    }

    if (own_deferred.empty() && hoists.empty()) {
        return false;
    }

    // ── 3. First-use index for parent's own deferred tensors ──────────────
    auto                                &nodes = graph.nodes();
    std::unordered_map<TensorId, size_t> first_use;
    for (size_t idx = 0; idx < nodes.size(); idx++) {
        for (auto tid : nodes[idx].inputs) {
            if (first_use.find(tid) == first_use.end()) {
                first_use[tid] = idx;
            }
        }
        for (auto tid : nodes[idx].outputs) {
            if (first_use.find(tid) == first_use.end()) {
                first_use[tid] = idx;
            }
        }
    }

    // ── 4. Build the insertion plan ───────────────────────────────────────
    struct Insertion {
        size_t            position;
        std::vector<Node> new_nodes;
    };
    std::vector<Insertion> insertions;

    for (auto tid : own_deferred) {
        auto &handle = graph.tensor(tid);

        size_t insert_pos = 0;
        if (auto it = first_use.find(tid); it != first_use.end()) {
            insert_pos = it->second;
        }

        auto new_nodes = build_lifecycle_nodes(handle, /*owns_tid=*/true);
        // Stats: count of Materialize + (Initialize if present).
        _num_materialized++;
        if (handle.init_kind != InitKind::None) {
            _num_initialized++;
        }

        bool const is_dist = handle.is_distributed && !handle.is_replicated;
        EINSUMS_LOG_INFO("Materialization: inserting materialize({}){} before position {}", handle.name, is_dist ? " [distributed]" : "",
                         insert_pos);

        insertions.push_back({.position = insert_pos, .new_nodes = std::move(new_nodes)});
    }

    for (auto const &h : hoists) {
        auto &handle = h.handle_owner->tensor(h.tid);

        auto new_nodes = build_lifecycle_nodes(handle, /*owns_tid=*/false);
        _num_materialized++;
        if (handle.init_kind != InitKind::None) {
            _num_initialized++;
        }

        EINSUMS_LOG_INFO("Materialization: hoisting materialize({}) from '{}' to position {}", handle.name, h.handle_owner->name(),
                         h.owning_node_index);

        insertions.push_back({.position = h.owning_node_index, .new_nodes = std::move(new_nodes)});
    }

    // ── 5. Apply, descending so earlier positions don't shift ────────────
    std::ranges::sort(insertions, [](Insertion const &a, Insertion const &b) { return a.position > b.position; });

    for (auto &ins : insertions) {
        nodes.insert(nodes.begin() + static_cast<ptrdiff_t>(ins.position), std::make_move_iterator(ins.new_nodes.begin()),
                     std::make_move_iterator(ins.new_nodes.end()));
    }

    graph.mark_sorted();

    return true;
}

} // namespace einsums::compute_graph::passes
