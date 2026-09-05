//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/DistributionDescriptor.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/Materialization.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

// Build the Materialize (and optional Initialize) node for a single
// deferred tensor. The returned vector either has size 1 (Materialize
// only) or 2 (Materialize + Initialize).
//
// @param handle         Mutable handle whose ``materialize_fn`` /
//                       ``init_kind`` / distribution metadata drives the
//                       node's execute closure.
// @param emit_tid       TensorId this graph resolves the buffer to; the
//                       synthetic nodes carry it as their output so the
//                       dependency builder orders the control-flow node
//                       (whose effective I/O maps the same buffer pointer
//                       to this id) after the Materialize / Initialize.
//                       For a hoisted body tensor this is a parent id
//                       minted via Graph::find_or_register_tensor_ptr,
//                       not the body's own id.
std::vector<Node> build_lifecycle_nodes(TensorHandle &handle, TensorId emit_tid) {
    std::vector<Node> out;

    // ── Materialize ────────────────────────────────────────────────────
    {
        Node mat_node;
        mat_node.kind    = OpKind::Materialize;
        mat_node.label   = fmt::format("materialize({})", handle.name);
        mat_node.outputs = {emit_tid};

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
        init_node.kind    = OpKind::Initialize;
        init_node.outputs = {emit_tid};

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

// One deferred tensor, paired with the graph whose registry owns its handle.
struct DeferredEntry {
    Graph   *handle_owner;
    TensorId tid;
};

// Appends every deferred tensor of @p graph and of every sub-graph nested under
// it (loop bodies, conditional branches, and their nesting), each paired with
// the graph that owns its handle, so the caller can build hoisted lifecycle
// nodes for it. @p graph's own tensors come first, then each sub-graph's in
// turn, depth first.
//
// A SETUP body met on the way is not descended into. Its workspace is materialized inside the
// body, whatever depth the body sits at, and the body is handed back through @p nested_setups so
// the setup-body arm can do that. Hoisting a setup's workspace to the outermost parent gave it a
// lifecycle the body's own validation could not see: a body checks its own nodes and its
// descendants for the Materialize, never its ancestors, and the parent-placed node had no edge to
// the loop the body sat in, so whether it had run first was a matter of schedule order. That
// order differed between platforms, which is how one program passed on three and failed on one.
void collect_deferred(Graph &graph, std::vector<DeferredEntry> &out, std::vector<std::pair<Graph *, std::string>> &nested_setups) {
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.alloc_state == AllocState::Deferred) {
            out.push_back({.handle_owner = &graph, .tid = tid});
        }
    }
    for (auto &node : graph.nodes()) {
        if (auto *loop = std::get_if<LoopDescriptor>(&node.op_data)) {
            if (loop->body) {
                collect_deferred(*loop->body, out, nested_setups);
            }
        } else if (auto *cond = std::get_if<ConditionalDescriptor>(&node.op_data)) {
            if (cond->then_branch) {
                collect_deferred(*cond->then_branch, out, nested_setups);
            }
            if (cond->else_branch) {
                collect_deferred(*cond->else_branch, out, nested_setups);
            }
        } else if (auto *setup = std::get_if<SetupDescriptor>(&node.op_data)) {
            if (setup->body) {
                nested_setups.emplace_back(setup->body.get(), node.label);
            }
        }
    }
}

/// Whether @p graph already holds a Materialize node for the tensor named @p name.
///
/// Cheap and name-keyed, matching what FreeInsertion does for the same question. The pass is
/// re-runnable (a manager may apply it twice, and the load path applies it to a graph a save
/// was taken before), so emitting a second lifecycle for a tensor that already has one has to
/// be impossible rather than merely unlikely.
bool already_materialized_in(Graph const &graph, std::string const &name) {
    std::string const want = fmt::format("materialize({})", name);
    return std::ranges::any_of(graph.nodes(), [&want](Node const &node) { return node.kind == OpKind::Materialize && node.label == want; });
}

/// What a Materialize node's label names, or empty for any other label.
std::string materialized_name(std::string const &label) {
    constexpr std::string_view prefix = "materialize(";
    if (!label.starts_with(prefix) || !label.ends_with(')')) {
        return {};
    }
    return label.substr(prefix.size(), label.size() - prefix.size() - 1);
}

/// One graph's contribution to the audit, then every sub-graph's.
///
/// The used and owned sets are keyed by tensor NAME. Ids are per graph, and a body's copy of a
/// parent's buffer carries the parent's name, which is the only thing that lets a hoisted lifecycle
/// and the use it exists for be recognized as being about one tensor.
///
/// The Materialize count is keyed by BUFFER where the node names one, and by name only where it
/// does not: two setup bodies may each declare a workspace of one name, as a fitting emitted into
/// a setup and again into a loop body does, and those are two tensors with one lifecycle each
/// rather than one tensor with two. The key carries the name so the report can still say it.
void collect_audit(Graph const &graph, std::map<std::string, std::size_t> &materialized, std::set<std::string> &used,
                   std::set<std::string> &owned_deferred) {
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.alloc_state == AllocState::Deferred && handle.is_intermediate) {
            owned_deferred.insert(handle.name);
        }
    }

    for (auto const &node : graph.nodes()) {
        if (node.kind == OpKind::Materialize) {
            if (std::string name = materialized_name(node.label); !name.empty()) {
                void const *buffer = nullptr;
                if (!node.outputs.empty()) {
                    if (TensorHandle const *handle = graph.find_tensor(node.outputs.front()); handle != nullptr) {
                        buffer = handle->tensor_ptr;
                    }
                }
                ++materialized[buffer != nullptr ? fmt::format("{}@{}", name, buffer) : name];
            }
            continue;
        }
        if (is_lifecycle(node.kind)) {
            continue;
        }
        // Through aliases: a tensor written only through a view of it is used, and the view is a
        // separate handle with a name of its own.
        auto note = [&](TensorId tid) {
            if (TensorHandle const *handle = graph.find_tensor(graph.resolve_alias(tid)); handle != nullptr) {
                used.insert(handle->name);
            }
        };
        for (TensorId const tid : node.inputs) {
            note(tid);
        }
        for (TensorId const tid : node.outputs) {
            note(tid);
        }
    }

    graph.for_each_subgraph([&](Graph const &sub) { collect_audit(sub, materialized, used, owned_deferred); });
}

} // namespace

std::vector<std::string> duplicate_materializations(Graph const &graph) {
    std::map<std::string, std::size_t> materialized;
    std::set<std::string>              used;
    std::set<std::string>              owned_deferred;
    collect_audit(graph, materialized, used, owned_deferred);

    std::vector<std::string> out;
    for (auto const &[key, count] : materialized) {
        if (count > 1) {
            out.push_back(key.substr(0, key.find('@')));
        }
    }
    return out;
}

std::vector<std::string> stranded_materializations(Graph const &graph) {
    std::map<std::string, std::size_t> materialized;
    std::set<std::string>              used;
    std::set<std::string>              owned_deferred;
    collect_audit(graph, materialized, used, owned_deferred);

    std::vector<std::string> out;
    for (auto const &[key, count] : materialized) {
        std::string const name = key.substr(0, key.find('@'));
        if (owned_deferred.contains(name) && !used.contains(name)) {
            out.push_back(name);
        }
    }
    return out;
}

std::vector<std::string> Materialization::explain() const {
    if (_num_materialized == 0 && _num_initialized == 0) {
        return {};
    }
    return {fmt::format("Materialization: allocated {} deferred tensor(s), zero-initialized {}", _num_materialized, _num_initialized)};
}

void Materialization::reset_stats() {
    _num_materialized = 0;
    _num_initialized  = 0;
    _num_unused       = 0;
}

bool Materialization::run(Graph &graph) {
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
    //      know which node index owns each sub-graph, that's where the
    //      hoisted Materialize / Initialize node goes.
    struct Hoist {
        size_t   owning_node_index;
        Graph   *handle_owner;
        TensorId tid;
    };
    std::vector<Hoist> hoists;
    // Setup bodies found under a loop or a branch: their workspace is placed inside them by the
    // setup-body arm rather than hoisted, see collect_deferred.
    std::vector<std::pair<Graph *, std::string>> nested_setups;

    auto const &parent_nodes = graph.nodes();
    for (size_t i = 0; i < parent_nodes.size(); i++) {
        Node const &node = parent_nodes[i];

        auto collect_from = [&](Graph &child) {
            std::vector<DeferredEntry> found;
            collect_deferred(child, found, nested_setups);
            for (auto const &e : found) {
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
    // Shared UsageAnalysis instead of a private scan. Two deliberate
    // semantic upgrades over the old raw loop: alias chains resolve (a view
    // of a deferred tensor counts as a use of the owner), and a use that
    // exists only inside a Loop/Conditional body surfaces at the
    // control-flow node's position (the Materialize then lands right before
    // the loop instead of defaulting to position 0).
    auto       &nodes = graph.nodes();
    auto const &ua    = graph.usage();

    // ── 4. Build the insertion plan ───────────────────────────────────────
    struct Insertion {
        size_t            position;
        std::vector<Node> new_nodes;
    };
    std::vector<Insertion> insertions;

    // A deferred tensor used both in the parent and inside a sub-graph (or in
    // more than one sub-graph) is ONE underlying buffer but shows up once in
    // own_deferred and again in hoists. It must be materialized + initialized
    // exactly ONCE, before its earliest use, emitting a lifecycle pair per
    // use-site re-runs Initialize (e.g. re-zeroes the buffer) and clobbers a
    // value an earlier use already produced (e.g. a loop's accumulation read by
    // a later parent op). Dedup by the underlying tensor_ptr, preferring a
    // parent-owning request so the node carries the parent TensorId (and thus
    // its dependency edges), placed at the earliest position across all uses.
    struct Req {
        size_t   position;
        bool     owns_tid;
        Graph   *owner;
        TensorId tid;
    };
    std::vector<Req>                         reqs;
    std::unordered_map<void const *, size_t> req_of_ptr;
    auto                                     add_req = [&](void const *ptr, size_t position, bool owns_tid, Graph *owner, TensorId tid) {
        // A null ptr can't be deduped reliably, keep it as its own request.
        if (ptr != nullptr) {
            if (auto it = req_of_ptr.find(ptr); it != req_of_ptr.end()) {
                Req &b     = reqs[it->second];
                b.position = std::min(b.position, position);
                if (owns_tid && !b.owns_tid) {
                    b.owns_tid = true;
                    b.owner    = owner;
                    b.tid      = tid;
                }
                return;
            }
            req_of_ptr.emplace(ptr, reqs.size());
        }
        reqs.push_back({position, owns_tid, owner, tid});
    };

    for (auto tid : own_deferred) {
        auto  &handle     = graph.tensor(tid);
        size_t insert_pos = 0;
        bool   used       = false;
        if (auto const *use = ua.find_owner(tid)) {
            if (size_t const fu = use->first_use(); fu != TensorUsage::npos) {
                insert_pos = fu;
                used       = true;
            }
        }
        // A graph-owned intermediate that no node reads or writes, here or in any sub-graph, has
        // no value to hold. A structural pass that dissolved the intermediate leaves its
        // declaration behind, since a caller may still hold the handle, and allocating storage
        // for it would spend memory on a tensor whose whole point was to stop existing: for the
        // CCSD tau terms that is a v^4 buffer nobody writes.
        if (!used && handle.is_intermediate) {
            _num_unused++;
            report(2, fmt::format("deferred tensor '{}' is used by no node; left unallocated", handle.name));
            continue;
        }
        // A tensor another pass already gave a lifecycle to is not this pass's to give a second
        // one. ContractionPlanning emits its own Materialize for the scratch it declares, so that
        // a standalone application of it produces an executable graph, and this pass then found
        // the same deferred declaration and emitted another. Two allocations of one buffer is
        // survivable only because materialize_fn happens to be idempotent, which is a property of
        // today's hook rather than a contract, and the node it adds carries an edge that serializes
        // the chain against itself for nothing. Name-keyed, matching what the setup-body arm above
        // already does through already_materialized_in and for the same reason.
        if (already_materialized_in(graph, handle.name)) {
            report(2, fmt::format("deferred tensor '{}' already has a Materialize node; left to it", handle.name));
            continue;
        }
        add_req(handle.tensor_ptr, insert_pos, /*owns_tid=*/true, &graph, tid);
    }
    for (auto const &h : hoists) {
        auto &handle = h.handle_owner->tensor(h.tid);
        add_req(handle.tensor_ptr, h.owning_node_index, /*owns_tid=*/false, h.handle_owner, h.tid);
    }

    // A buffer whose first use is a Setup node that WRITES it is produced once per bound
    // problem, and its lifecycle belongs on the same schedule. Left in the parent, the
    // Initialize runs on every replay and zeroes a fitting the skipped body will not
    // recompute, which is a silently wrong answer on the second replay and a correct one on
    // the first. So the pair goes to the front of the setup body instead, where it runs
    // exactly when the thing it prepares storage for runs.
    //
    // The setup may sit BELOW the node at that position rather than be it: a fitting emitted into
    // a loop body puts its setup inside the loop, and the parent holds a handle for that setup's
    // workspace whose first use is the Loop node. Looking only at the node itself sent such a
    // workspace to a parent-level lifecycle the nested body's validation could not see, and
    // whether execute reached the body before or after that lifecycle was a matter of schedule
    // order. So the search descends through loops and branches to the setup that writes the
    // buffer, at any depth.
    auto body_writes = [](Graph &body, void const *ptr) {
        for (auto const &node : body.nodes()) {
            for (TensorId const out : node.outputs) {
                TensorHandle const *written = body.find_tensor(body.resolve_alias(out));
                if (written != nullptr && written->tensor_ptr == ptr) {
                    return true;
                }
            }
        }
        return false;
    };
    // NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest, so the search does too.
    std::function<Graph *(Node const &, void const *)> setup_below = [&](Node const &node, void const *ptr) -> Graph * {
        if (auto const *desc = std::get_if<SetupDescriptor>(&node.op_data)) {
            return (desc->body && body_writes(*desc->body, ptr)) ? desc->body.get() : nullptr;
        }
        auto search = [&](Graph &child) -> Graph * {
            for (auto const &inner : child.nodes()) {
                if (Graph *found = setup_below(inner, ptr); found != nullptr) {
                    return found;
                }
            }
            return nullptr;
        };
        if (auto const *loop = std::get_if<LoopDescriptor>(&node.op_data)) {
            return loop->body ? search(*loop->body) : nullptr;
        }
        if (auto const *cond = std::get_if<ConditionalDescriptor>(&node.op_data)) {
            if (cond->then_branch) {
                if (Graph *found = search(*cond->then_branch); found != nullptr) {
                    return found;
                }
            }
            return cond->else_branch ? search(*cond->else_branch) : nullptr;
        }
        return nullptr;
    };
    //
    // The setup is found by SCANNING the node list in order rather than by looking at the node at
    // the buffer's recorded first use. A setup node carries no inputs or outputs of its own and
    // orders itself against its neighbours through effective io alone, so two setups that share
    // no tensor are free to be reordered, and a first-use position taken before such a reorder
    // names whichever setup now sits there. On one platform that happened to be the setup that
    // writes the buffer and on another it was not, and the buffer's lifecycle then went to a node
    // the writing body's validation could not see. The earliest setup that writes the buffer is a
    // property of the graph, not of a position, so it is what is looked for.
    auto setup_body_writing = [&](std::size_t /*position*/, void const *ptr) -> Graph * {
        if (ptr == nullptr) {
            return nullptr;
        }
        for (auto const &node : nodes) {
            if (Graph *found = setup_below(node, ptr); found != nullptr) {
                return found;
            }
        }
        return nullptr;
    };

    /// The body's own id for the buffer @p ptr names, which is what a node placed in the
    /// body has to carry: ids are per-graph and the parent's mean nothing there.
    auto body_tid_for = [](Graph &body, void const *ptr) -> std::optional<TensorId> {
        TensorId const tid = body.find_tensor_id_by_ptr(ptr);
        return tid != 0 ? std::optional<TensorId>{tid} : std::nullopt;
    };

    // build_lifecycle_nodes plus the two counters that always move with it: every
    // lifecycle this pass emits materializes one tensor, and initializes it as well
    // when the handle asks for it. Kept together so the three cannot drift.
    auto lifecycle_for = [this](TensorHandle &handle, TensorId emit_tid) {
        auto built = build_lifecycle_nodes(handle, emit_tid);
        _num_materialized++;
        if (handle.init_kind != InitKind::None) {
            _num_initialized++;
        }
        return built;
    };

    // Splice a lifecycle to the front of a sub-graph, which is where it has to run:
    // the body's own nodes are all consumers of it.
    auto insert_at_front = [](Graph &target, std::vector<Node> lifecycle) {
        std::vector<std::pair<std::size_t, std::vector<Node>>> group;
        group.emplace_back(0, std::move(lifecycle));
        target.insert_node_groups(std::move(group));
    };

    bool modified = false;

    // Every tensor the request loop places a lifecycle for, by IDENTITY, so the setup-body arm
    // below does not place a second one. Its own guard looks inside the body it is walking and is
    // blind to a node the request loop put in the PARENT, which is where a tensor declared in a
    // loop body gets one: a fitting that is emitted both into a setup and into a loop body, which
    // is what a re-fitted amplitude is, presents exactly that shape.
    //
    // Keyed on the buffer rather than on the name, because that same re-fitted amplitude declares
    // the same workspace NAMES in two different bodies, and a name-keyed set let whichever body
    // the request loop reached first claim the name for both: the other body's workspace was then
    // skipped by the arm below and reached execute unallocated. Which body came first depended on
    // iteration order, so the same program passed on one platform and failed on the next.
    std::set<void const *> placed;

    for (auto const &r : reqs) {
        auto &handle = r.owner->tensor(r.tid);
        if (handle.tensor_ptr != nullptr) {
            placed.insert(handle.tensor_ptr);
        }

        if (Graph *body = setup_body_writing(r.position, handle.tensor_ptr); body != nullptr) {
            if (auto const body_tid = body_tid_for(*body, handle.tensor_ptr); body_tid.has_value()) {
                auto body_nodes = lifecycle_for(handle, *body_tid);
                report(2, fmt::format("materialize deferred tensor '{}' inside setup body '{}'", handle.name, nodes[r.position].label));
                insert_at_front(*body, std::move(body_nodes));
                continue;
            }
        }

        // A parent-owned request emits its own id; a hoisted body request
        // resolves the buffer to a parent id (minting one if effective_io
        // has not yet), so the Loop / Conditional node gets a RAW edge after
        // these lifecycle nodes instead of floating as an edgeless root.
        TensorId const emit_tid  = r.owns_tid ? r.tid : graph.find_or_register_tensor_ptr(handle);
        auto           new_nodes = lifecycle_for(handle, emit_tid);
        EINSUMS_LOG_INFO("Materialization: materialize({}) at position {} (owns_tid={})", handle.name, r.position, r.owns_tid);
        report(2, fmt::format("materialize deferred tensor '{}' at position {}{}", handle.name, r.position,
                              handle.init_kind != InitKind::None ? " (+initialize)" : ""));
        insertions.push_back({.position = r.position, .new_nodes = std::move(new_nodes)});
    }

    // A setup body's OWN workspace, materialized inside the body. The parent cannot see these
    // through either path above: they are not its tensors, and the hoist walk deliberately
    // covers only loop bodies and conditional branches, where a hoisted lifecycle is what
    // stops an allocation happening per iteration.
    //
    // AFTER the request loop, deliberately, and the ordering is a fix rather than a tidy-up.
    // The comment below says a body's copy of a parent-declared tensor carries no allocating
    // hook and is therefore skipped here; that is false for a deferred runtime tensor, which
    // capture adopts into the body complete with one. So a setup output the parent declared
    // got a lifecycle from BOTH arms, and every graph with a setup body has carried two
    // Materialize nodes per fitting since setup bodies shipped. Running second means the
    // name-keyed guard below sees the request loop's node, which is the one built from the
    // PARENT's handle and is the one that has to win: it allocates the buffer the parent's
    // readers hold.
    //
    // Inside rather than hoisted, because a setup body runs once per bound problem and is
    // skipped by every replay after. A parent-placed Materialize would allocate the fitting's
    // scratch on every replay to feed a body that does not run.
    //
    // The provider used to do this itself, by applying a pass manager to the body before
    // handing it over. That was wrong for a reason worth keeping written down: a Materialize
    // node carries an allocating closure, a closure is the one thing a file cannot hold, and
    // allocation is a resource decision that the design says is re-derived on load rather than
    // saved. Doing it at capture baked a resource decision into structure and made the fitting
    // unsaveable, which is the one thing a factorization exists to avoid.
    std::vector<std::pair<Graph *, std::string>> setup_bodies;
    for (auto const &node : nodes) {
        if (auto const *setup = std::get_if<SetupDescriptor>(&node.op_data); setup != nullptr && setup->body) {
            setup_bodies.emplace_back(setup->body.get(), node.label);
        }
    }
    setup_bodies.insert(setup_bodies.end(), nested_setups.begin(), nested_setups.end());
    for (auto const &[body_ptr, setup_label] : setup_bodies) {
        Graph &body = *body_ptr;

        std::vector<Node> body_lifecycle;
        for (auto &[tid, handle] : body.tensors_map()) {
            if (handle.alloc_state != AllocState::Deferred) {
                continue;
            }
            // A handle with no allocating hook is not this graph's to allocate. That is
            // exactly the body's copy of a tensor the PARENT declared, which the block above
            // has already placed a Materialize for; emitting a second one here would double
            // the node and, since the copy carries no hook, the second would allocate nothing.
            if (!handle.materialize_fn) {
                continue;
            }
            if ((handle.tensor_ptr != nullptr && placed.contains(handle.tensor_ptr)) || already_materialized_in(body, handle.name)) {
                continue;
            }
            for (auto &node : lifecycle_for(handle, tid)) {
                body_lifecycle.push_back(std::move(node));
            }
            report(2, fmt::format("materialize setup-body scratch '{}' inside '{}'", handle.name, setup_label));
        }
        if (!body_lifecycle.empty()) {
            insert_at_front(body, std::move(body_lifecycle));
            modified = true;
        }
    }

    // ── 5. Apply all insertions (Graph orders them descending and re-sorts) ─
    std::vector<std::pair<std::size_t, std::vector<Node>>> groups;
    groups.reserve(insertions.size());
    for (auto &ins : insertions) {
        groups.emplace_back(ins.position, std::move(ins.new_nodes));
    }
    if (!groups.empty()) {
        graph.insert_node_groups(std::move(groups));
        modified = true;
    }

    report(1, fmt::format("materialized {} deferred tensor(s) ({} initialized, {} unused and left unallocated)", _num_materialized,
                          _num_initialized, _num_unused));
    return modified;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
