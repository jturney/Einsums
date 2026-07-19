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
//            False for hoisted frees, the tid belongs to a child graph,
//            so we leave the Free's inputs empty and rely on position
//            (FreeInsertion runs near the end of the pipeline and marks
//            the graph sorted, so nothing reorders it).
struct FreePlan {
    size_t   position;
    Graph   *owner;
    TensorId tid;
    bool     owns_tid;

    /// Insert a paired Materialize BEFORE this parent-node index, or SIZE_MAX
    /// for none. Deferred tensors already got a Materialize node from the
    /// Materialization pass; eager create_tensor intermediates did not, so a
    /// bare Free left the second execute() reading released storage (the
    /// buffer is reclaimed by release_fn but nothing reallocates it).
    size_t materialize_before{SIZE_MAX};
};

// Is `name` already freed by an existing Free node anywhere in `nodes`?
// Used for idempotency across repeated pass runs. Label-based because a
// hoisted Free doesn't carry the (foreign) child tid as an input.
bool already_has_free_named(std::vector<Node> const &nodes, std::string const &name) {
    auto const want = fmt::format("free({})", name);
    return std::ranges::any_of(nodes, [&](Node const &n) { return n.kind == OpKind::Free && n.label == want; });
}

// Same idempotency check for the paired Materialize (also label-based: the
// Materialization pass and this pass both label materialize(<name>)).
bool already_has_materialize_named(std::vector<Node> const &nodes, std::string const &name) {
    auto const want = fmt::format("materialize({})", name);
    return std::ranges::any_of(nodes, [&](Node const &n) { return n.kind == OpKind::Materialize && n.label == want; });
}

// Collect freeable intermediates from every descendant of `graph` (loop
// bodies, conditional branches, and any nesting underneath), in
// post-order. A body-scoped intermediate is live across every iteration
// of the loop that contains it, so its single Free belongs in the parent
// *after* the outermost enclosing loop, never inside the body, which
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

    // ── Part A: parent-level intermediates ────────────────────────────────
    // Free each intermediate right after its last real use. Positions come
    // from the shared UsageAnalysis (owner-resolved, one scan for the whole
    // pipeline) with this pass's policy filters: a Free node's input is a
    // scheduling edge, not a use (keeps the pass idempotent across repeated
    // runs), and Alloc marks creation, not a data write - the paired
    // Materialize must precede the first REAL writer so a replayed graph
    // reallocates before producing into the buffer. Subtree expansion is ON
    // for the last use: a PARENT-declared intermediate consumed only inside
    // a loop body has its real last use at the Loop node, and without the
    // subtree view Part A freed it right after its Initialize - before the
    // loop ever ran. (Part B cannot cover that case: the body-map handles of
    // parent-declared tensors carry neither is_intermediate nor release_fn,
    // so its collect filter never sees them.) Overlap with Part B for
    // body-CREATED buffers is prevented by the planned-name dedup below.
    auto const &ua = graph.usage();

    std::vector<FreePlan>           plans;
    std::unordered_set<std::string> planned_names;

    for (auto const &[tid, use] : ua.table()) {
        size_t const last_idx = use.last_use(/*ignore_free=*/true, /*include_subtree=*/true);
        if (last_idx == TensorUsage::npos)
            continue;
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

        size_t       mat_before = SIZE_MAX;
        size_t const fw         = use.first_writer(/*ignore_alloc=*/true, /*include_subtree=*/false);
        if (fw != TensorUsage::npos && handle.materialize_fn && !already_has_materialize_named(nodes, handle.name)) {
            mat_before = fw;
        }

        plans.push_back({.position = last_idx, .owner = &graph, .tid = tid, .owns_tid = true, .materialize_before = mat_before});
        planned_names.insert(handle.name);
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
                if (already_has_free_named(nodes, handle.name) || planned_names.contains(handle.name)) {
                    continue;
                }
                size_t mat_before = SIZE_MAX;
                if (handle.materialize_fn && !already_has_materialize_named(nodes, handle.name) &&
                    !already_has_materialize_named(f.owner->nodes(), handle.name)) {
                    mat_before = i;
                }
                plans.push_back({.position = i, .owner = f.owner, .tid = f.tid, .owns_tid = false, .materialize_before = mat_before});
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

    // Build every insertion (Frees and their paired Materializes), then
    // apply them in descending index order so earlier positions stay valid.
    struct Insertion {
        size_t index;
        Node   node;
    };
    std::vector<Insertion> insertions;
    insertions.reserve(plans.size() * 2);

    for (auto const &plan : plans) {
        auto &handle = plan.owner->tensor(plan.tid);

        auto         rel_fn = handle.release_fn;
        size_t const bytes  = handle.total_bytes();

        Node free_node;
        free_node.kind  = OpKind::Free;
        free_node.label = fmt::format("free({})", handle.name);
        if (plan.owns_tid) {
            // The tensor is BOTH an input and an output. The input edge orders
            // the Free after the last writer; the output makes the Free a
            // writer itself, so the dependency builder's WAR scan orders it
            // after every prior READER too. With only the input, a concurrent
            // executor sees the Free as just another reader and can release
            // the buffer while a real consumer is still reading it (the
            // serial executor was safe only by node position).
            free_node.inputs  = {plan.tid};
            free_node.outputs = {plan.tid};
        }

        free_node.execute = [rel_fn, bytes, name = handle.name]() {
            if (rel_fn) {
                rel_fn();
                EINSUMS_LOG_DEBUG("FreeInsertion: released '{}' ({} bytes)", name, bytes);
            }
        };
        free_node.estimated_bytes = bytes;

        report(2, fmt::format("insert Free for '{}' ({} bytes) after its last consumer at position {}", handle.name, bytes, plan.position));
        insertions.push_back({.index = plan.position + 1, .node = std::move(free_node)});
        _num_freed++;

        if (plan.materialize_before != SIZE_MAX) {
            Node mat_node;
            mat_node.kind  = OpKind::Materialize;
            mat_node.label = fmt::format("materialize({})", handle.name);
            if (plan.owns_tid) {
                mat_node.outputs = {plan.tid}; // WAW edge orders it before the writer.
            }

            mat_node.execute = [mat_fn = handle.materialize_fn]() {
                // Idempotent: a no-op on the first execute (the eager tensor
                // is already allocated); reallocates on every replay after
                // the Free above reclaimed the buffer.
                mat_fn();
            };
            mat_node.estimated_bytes = bytes;

            report(2, fmt::format("pair Materialize for eager '{}' before its first writer at position {}", handle.name,
                                  plan.materialize_before));
            insertions.push_back({.index = plan.materialize_before, .node = std::move(mat_node)});
        }
    }

    std::ranges::sort(insertions, [](Insertion const &a, Insertion const &b) { return a.index > b.index; });
    for (auto &ins : insertions) {
        nodes.insert(nodes.begin() + static_cast<ptrdiff_t>(ins.index), std::move(ins.node));
    }

    if (_num_freed > 0) {
        graph.mark_sorted();
        EINSUMS_LOG_INFO("FreeInsertion: inserted {} Free nodes", _num_freed);
        report(1, fmt::format("inserted {} Free node(s) to cap peak memory", _num_freed));
    }

    return _num_freed > 0;
}

} // namespace einsums::compute_graph::passes
