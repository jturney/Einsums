//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/FreeInsertion.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

// A Free we plan to insert into the *parent* graph.
//
// position : insert AFTER this parent-node index.
// owner    : graph whose tensor_map holds the handle (parent for flat
//            frees; a descendant body/branch for hoisted frees).
// tid      : tensor id within `owner`.
// owns_tid : true when `tid` lives in the graph being mutated, so the
//            Free node can carry it as an input for dependency tracking.
//            False for hoisted frees — the tid belongs to a child graph,
//            so we leave the Free's inputs empty and rely on position
//            (FreeInsertion runs near the end of the pipeline and marks
//            the graph sorted, so nothing reorders it).
struct FreePlan {
    size_t   position;
    Graph   *owner;
    TensorId tid;
    bool     owns_tid;
};

// Is `name` already freed by an existing Free node anywhere in `nodes`?
// Used for idempotency across repeated pass runs. Label-based because a
// hoisted Free doesn't carry the (foreign) child tid as an input.
bool already_has_free_named(std::vector<Node> const &nodes, std::string const &name) {
    auto const want = fmt::format("free({})", name);
    return std::ranges::any_of(nodes, [&](Node const &n) { return n.kind == OpKind::Free && n.label == want; });
}

// Collect freeable intermediates from every descendant of `graph` (loop
// bodies, conditional branches, and any nesting underneath), in
// post-order. A body-scoped intermediate is live across every iteration
// of the loop that contains it, so its single Free belongs in the parent
// *after* the outermost enclosing loop — never inside the body, which
// would free-then-reuse each iteration.
struct DescendantFreeable {
    Graph   *owner;
    TensorId tid;
};

void collect_descendant_freeable(Graph &graph, size_t min_bytes, std::vector<DescendantFreeable> &out) {
    graph.for_each_subgraph([&](Graph &sub) {
        for (auto const &[tid, handle] : sub.tensors_map()) {
            if (handle.is_intermediate && handle.release_fn && handle.aliases == 0 && handle.total_bytes() >= min_bytes) {
                out.push_back({.owner = &sub, .tid = tid});
            }
        }
        collect_descendant_freeable(sub, min_bytes, out);
    });
}

} // namespace

bool FreeInsertion::run(Graph &graph) {
    _num_freed = 0;

    auto       &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    if (nodes.empty())
        return false;

    // Resolve a TensorId through any chain of aliases to the underlying owner.
    // Aliases (View outputs) read/write through the parent's storage, so a use
    // of the alias is logically a use of the owner — extend the owner's
    // lifetime accordingly.
    auto resolve_owner = [&](TensorId id) {
        for (int hops = 0; hops < 32; ++hops) {
            auto it = tensors.find(id);
            if (it == tensors.end() || it->second.aliases == 0)
                return id;
            id = it->second.aliases;
        }
        return id;
    };

    // ── Part A: parent-level intermediates ────────────────────────────────
    // Find the last node that references each intermediate, then free it
    // right after that node. Unchanged from the original flat-graph logic.
    std::unordered_map<TensorId, size_t> last_use;

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        // A Free node carries the freed tensor as an input purely for
        // dependency ordering — it isn't a real "use" that extends the
        // tensor's lifetime. Skipping it keeps the pass idempotent: on a
        // repeated run, last_use stays at the genuine final consumer, so
        // the forward dedup scan below still finds the existing Free.
        if (nodes[idx].kind == OpKind::Free) {
            continue;
        }
        for (auto tid : nodes[idx].inputs) {
            TensorId const owner = resolve_owner(tid);
            auto           it    = tensors.find(owner);
            if (it != tensors.end() && it->second.is_intermediate) {
                last_use[owner] = idx;
            }
        }
        for (auto tid : nodes[idx].outputs) {
            TensorId const owner = resolve_owner(tid);
            auto           it    = tensors.find(owner);
            if (it != tensors.end() && it->second.is_intermediate) {
                last_use[owner] = idx;
            }
        }
    }

    std::vector<FreePlan> plans;

    for (auto const &[tid, last_idx] : last_use) {
        auto it = tensors.find(tid);
        if (it == tensors.end())
            continue;

        auto const &handle = it->second;

        if (!handle.is_intermediate || !handle.release_fn)
            continue;
        if (handle.total_bytes() < _min_bytes)
            continue;
        // Aliasing tensors (Views) don't own their storage.
        if (handle.aliases != 0)
            continue;

        // Don't double-free across repeated runs.
        bool already_freed = false;
        for (size_t idx = last_idx + 1; idx < nodes.size(); idx++) {
            if (nodes[idx].kind == OpKind::Free) {
                for (auto in_tid : nodes[idx].inputs) {
                    if (in_tid == tid) {
                        already_freed = true;
                        break;
                    }
                }
            }
            if (already_freed)
                break;
        }
        if (already_freed)
            continue;

        plans.push_back({.position = last_idx, .owner = &graph, .tid = tid, .owns_tid = true});
    }

    // ── Part B: body-resident intermediates, hoisted to after the loop ────
    // For each Loop / Conditional node in the parent, every freeable
    // intermediate reachable inside it (at any nesting depth) gets a single
    // Free emitted in the parent immediately after the owning node.
    for (size_t i = 0; i < nodes.size(); i++) {
        Node const &node = nodes[i];

        auto collect_from = [&](Graph &child) {
            std::vector<DescendantFreeable> found;
            for (auto const &[tid, handle] : child.tensors_map()) {
                if (handle.is_intermediate && handle.release_fn && handle.aliases == 0 && handle.total_bytes() >= _min_bytes) {
                    found.push_back({.owner = &child, .tid = tid});
                }
            }
            collect_descendant_freeable(child, _min_bytes, found);

            for (auto const &f : found) {
                auto const &handle = f.owner->tensor(f.tid);
                if (already_has_free_named(nodes, handle.name)) {
                    continue;
                }
                plans.push_back({.position = i, .owner = f.owner, .tid = f.tid, .owns_tid = false});
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

    if (plans.empty())
        return false;

    // Sort by position descending so inserts don't shift earlier positions.
    std::ranges::sort(plans, [](FreePlan const &a, FreePlan const &b) { return a.position > b.position; });

    for (auto const &plan : plans) {
        auto &handle = plan.owner->tensor(plan.tid);

        auto         rel_fn = handle.release_fn;
        size_t const bytes  = handle.total_bytes();

        Node free_node;
        free_node.kind  = OpKind::Free;
        free_node.label = fmt::format("free({})", handle.name);
        if (plan.owns_tid) {
            free_node.inputs = {plan.tid}; // Dependency: runs after last consumer.
        }
        free_node.outputs = {};

        free_node.execute = [rel_fn, bytes, name = handle.name]() {
            if (rel_fn) {
                rel_fn();
                EINSUMS_LOG_DEBUG("FreeInsertion: released '{}' ({} bytes)", name, bytes);
            }
        };
        free_node.estimated_bytes = bytes;

        nodes.insert(nodes.begin() + static_cast<ptrdiff_t>(plan.position + 1), std::move(free_node));
        _num_freed++;
    }

    if (_num_freed > 0) {
        graph.mark_sorted();
        EINSUMS_LOG_INFO("FreeInsertion: inserted {} Free nodes", _num_freed);
    }

    return _num_freed > 0;
}

} // namespace einsums::compute_graph::passes
