//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Execute.cpp
/// @brief Replay: running a captured graph, eagerly or through an executor.
///
/// @ref Graph::execute is the serial walk and the reference semantics; the
/// @ref Executor overload hands the same schedule to a parallel backend. Both
/// replay a schedule `Schedule.cpp` built, which is why neither computes an
/// order of its own.
///
/// What running one node means is `NodeRunner.hpp`'s. The optimization entry points sit here
/// too, because `optimize` is the other thing a caller does with a captured graph
/// and it is defined by what a later `execute` will see.

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/NodeRunner.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp> // For OptimizerPass and PassManager
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/ThreadPlanning.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Profile/Recording.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <ostream>
#include <queue>
#include <ranges>
#include <set>
#include <span>
#include <unordered_set>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// Leaves the replay a run context was entered for, however the replay ends.
struct LeaveReplay {
    detail::RunContext &context;
    explicit LeaveReplay(detail::RunContext &entered) : context(entered) {}
    LeaveReplay(LeaveReplay const &)            = delete;
    LeaveReplay &operator=(LeaveReplay const &) = delete;
    ~LeaveReplay() { context.leave(); }
};

} // namespace

namespace {

// The call sites every graph's zones report. The NAMES are per graph and per node and come
// pre-interned from _profile_strings; a site supplies the file, function and line, the same every
// time. Accessors, not function-local statics in execute(): a static there is built, and the
// profiler with it, even by a replay that records nothing.
waggle::ZoneSite const &graph_exec_site() {
    static waggle::ZoneSite const site{"ComputeGraph::execute", __FILE__, __LINE__, "execute", WAGGLE_CURRENT_DOMAIN};
    return site;
}

/// Serializes every stand-in re-seat in the process. One lock rather than one per graph,
/// because a stand-in is shared by every graph that adopted the tensor; a re-seat is a
/// comparison per slot, so nothing waits on it for long.
std::mutex &resync_mutex() {
    static std::mutex mutex;
    return mutex;
}

} // namespace

void Graph::resync_slot_storage() {
    // Slots carrying an adopted stand-in, and only those. A slot without one
    // addresses the CALLER's wrapper, or a tensor this graph created, and that
    // is the object Materialize and Free act through - it re-seats itself and
    // was never the stale one. Two reasons not to touch it anyway: the write
    // below would land on an object eager code may be using from another
    // thread, and a slot that did not adopt keeps the older "must outlive the
    // graph" contract, so its pointer is only as good as the node that reads it
    // needs it to be.
    //
    // A slot a redirect points at a stand-in is reached through the stand-in's
    // OWN slot, which is where the adoption is recorded; the object is one
    // object, so fixing it once fixes every slot naming it.
    //
    // Only the slots this graph's nodes name. A body replays inside its loop
    // node, and the parent orders that node against what the body's nodes
    // touch and nothing else. A stand-in the body adopted at capture can
    // outlive every node of the body that read it, when LoopInvariantHoisting
    // lifts them out; re-seating it read the storage block of a tensor that
    // a Free on another branch of the parent was releasing at the same time.
    // A sub-graph's nodes are re-seated by its own execute, through its own
    // slots.
    //
    // Each id once: the nodes name a few thousand slots between them, most of
    // them many times over, and re-seating one twice changes nothing.
    std::scoped_lock const lock(resync_mutex());
    if (!_resync_ids_valid || _resync_ids_version != _analysis_version || _resync_ids_nodes != _nodes.size()) {
        std::unordered_set<TensorId> seen;
        _resync_ids.clear();
        for (Node const &node : _nodes) {
            for (auto const *ids : {&node.inputs, &node.outputs}) {
                for (TensorId const id : *ids) {
                    if (seen.insert(id).second) {
                        _resync_ids.push_back(id);
                    }
                }
            }
        }
        _resync_ids_version = _analysis_version;
        _resync_ids_nodes   = _nodes.size();
        _resync_ids_valid   = true;
    }
    for (TensorId const id : _resync_ids) {
        resync_slot(id);
    }
}

void Graph::resync_slot_storage(std::span<TensorId const> ids) {
    std::scoped_lock const lock(resync_mutex());
    for (TensorId const id : ids) {
        resync_slot(id);
    }
}

void Graph::resync_slot(TensorId id) {
    if (auto const redirect = _slot_redirects.find(id); redirect != _slot_redirects.end()) {
        id = redirect->second;
    }
    TensorSlot const *slot = find_slot(id);
    if (slot != nullptr && slot->owner && slot->ptr != nullptr && slot->resync_of != nullptr) {
        slot->resync_of(slot->ptr);
    }
}

void Graph::lease_scratch() {
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
            einsums::detail::ScratchLease{.graph = _name, .state = _run_context, .replaying = [](void const *state) noexcept {
                                              return static_cast<detail::RunContext const *>(state)->replaying();
                                          }});
    }
    _scratch_leased_below = _next_tensor_id;
}

void Graph::execute() {
    // An installed executor (set_executor) takes over the whole run. This is
    // how loop bodies get a parallel backend: the loop node replays its body
    // via this argument-less execute(). Otherwise nodes run one at a time, in
    // the schedule's order.
    if (_executor) {
        execute(*_executor);
        return;
    }
    // Stateless, so one instance serves every graph on every thread.
    static SequentialExecutor sequential;
    execute(sequential);
}

void Graph::execute(Executor &executor) {
    lease_scratch();
    // An operand a Free released and a later Materialize replaced has moved,
    // and any stand-in this graph adopted for it still names the old buffer.
    // Before the alias derivation, which reads those addresses.
    resync_slot_storage();
    // Storage-level aliasing must be resolved before anything reasons about
    // which buffer a node touches; cheap and idempotent after the first call.
    link_alias_storage();
    replay(executor);
}

void Graph::replay(Executor &executor) {
    // Before the wall clock starts, so a replay that times re-planned widths
    // does not also time the planner.
    begin_replay_thread_plan();

    // Wall clock of this replay, for the thread-plan trial: replays are the
    // only thing the trial's two sides can be compared on.
    auto const replay_t0 = std::chrono::steady_clock::now();

    // The run metrics describe the outermost replay of the tree; a body's replay adds its node
    // times to it. Left however the replay ends, so a throw cannot leave the tree marked as
    // inside a replay.
    bool const        outermost = _run_context->enter();
    LeaveReplay const leave{*_run_context};
    if (outermost) {
        begin_run_metrics();
        detail::notify_replay(*_run_context, EventKind::ReplayStart, *this, nullptr);
    }

    // Rebuild when the order is unknown OR a pass vouched for the order via
    // mark_sorted() but left the position-keyed _deps stale (_deps_valid
    // false). topological_sort() takes the cheap rebuild_deps path in that
    // second case, keeping the pass-chosen order. Concurrent executors read
    // _deps directly, so stale lists would drop storage-reuse and lifecycle
    // edges a serial run never needed.
    if (!deps_current()) {
        topological_sort();
    }

    // Validate slot pointers once per change to the slot table (cheap check).
    // This catches cross-pipeline tensor misuse before it segfaults. Nothing
    // can invalidate a pointer between two replays of an unchanged graph, so
    // the walk is skipped on the replays that iterative workloads spend their
    // time in; every slot create/rebind/redirect and every pass run clears the
    // flag.
    if (!_slots_validated) {
        for (auto const &[id, slot] : _slot_map) {
            if (!slot)
                continue;
            if (slot->ptr == nullptr || reinterpret_cast<uintptr_t>(slot->ptr) < 4096) {
                std::string tname = slot->name.empty() ? fmt::format("id={}", id) : slot->name;
                EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                        "Graph '{}': tensor slot '{}' has invalid pointer (0x{:x}). "
                                        "This usually means the tensor was declared on a different pipeline/graph "
                                        "and wasn't properly shared via the workspace. "
                                        "Declare shared tensors on the Workspace, not on individual Pipelines.",
                                        _name, tname, reinterpret_cast<uintptr_t>(slot->ptr));
            }
        }
        _slots_validated = true;
    }

    if (!_executed) {
        materialize_internal_tensors();
        auto validation = validate_tensors();
        if (!validation) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", validation.error().message);
        }
        register_graph(this);
    }

    clear_timing_report();
    _timing_report_valid = false; // samples are about to be written

    // Read once so the zone pushes and their matching pops agree even if
    // another thread flips recording mid-run: an unbalanced zone deepens the
    // profiler tree without bound.
    bool const recording = profile::recording();

    // All fmt::format work for zones/annotations is precomputed, and so is the
    // string interning behind them; replays of an unchanged graph only pay the
    // profiler's event write. A run that records nothing builds none of it.
    if (recording && (!_profile_strings_valid || _profile_strings.size() != _nodes.size())) {
        rebuild_profile_strings();
    }

    {
        // RAII: pop the profiler zone even if the executor propagates an
        // exception. A bare push/pop here leaks an unclosed zone on every
        // failed execute, deepening the profiler tree without bound (which
        // makes the profiler->viewer serialization progressively slower).
        std::optional<waggle::ScopedZone> exec_zone;
        if (recording) {
            exec_zone.emplace(graph_exec_site(), _exec_zone_id);
            waggle::annotate("executor", executor.name());
        }

        // The executor chooses the order and the threads; the runner is what running each node
        // means, the same under every executor.
        NodeRunner runner(*this, recording);
        executor.execute(*this, runner);
        runner.finish();
    }
    _executed = true;

    // The safe point for the thread-plan trial (see plan_threads): the replay
    // has returned, so nothing is reading a width; every nested body replay it
    // started has returned with it; and this run's timings are complete and
    // recorded, which is the whole reason to wait for it.
    double const replay_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - replay_t0).count();
    contribute_run_samples(outermost);
    if (outermost) {
        end_run_metrics(replay_ms);
        if (_run_context->observer() != nullptr) {
            detail::notify_replay(*_run_context, EventKind::ReplayEnd, *this, &last_run());
        }
    }
    finish_replay_thread_plan(replay_ms);
}

// NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest, so the stamp does too.
std::uint64_t Graph::subtree_stamp() {
    if (_subgraph_positions_version != _analysis_version || _subgraph_positions_count != _nodes.size()) {
        _subgraph_positions.clear();
        for (std::size_t i = 0; i < _nodes.size(); ++i) {
            bool holds = false;
            for_each_child_graph(_nodes[i], [&holds](Graph &) { holds = true; });
            if (holds) {
                _subgraph_positions.push_back(static_cast<std::uint32_t>(i));
            }
        }
        _subgraph_positions_version = _analysis_version;
        _subgraph_positions_count   = _nodes.size();
    }

    // FNV-1a over (version, node count, its own subtree) per child, in walk order, so a body
    // edited back to an earlier count still moves the stamp through its version.
    std::uint64_t stamp = 14695981039346656037ULL;
    auto const    mix   = [&stamp](std::uint64_t value) { stamp = (stamp ^ value) * 1099511628211ULL; };
    for (std::uint32_t const pos : _subgraph_positions) {
        for_each_child_graph(_nodes[pos], [&mix](Graph &child) {
            mix(child._analysis_version);
            mix(child._nodes.size());
            mix(child.subtree_stamp());
        });
    }
    return stamp;
}

bool Graph::deps_current() {
    return _sorted && _deps_valid && _deps_subtree_stamp == subtree_stamp();
}

UsageAnalysis const &Graph::usage() {
    std::uint64_t const stamp = subtree_stamp();
    if (_usage_version != _analysis_version || _usage.node_count() != _nodes.size() || _usage_subtree_stamp != stamp) {
        _usage               = UsageAnalysis::build(*this);
        _usage_version       = _analysis_version;
        _usage_subtree_stamp = stamp;
    }
    return _usage;
}

bool Graph::apply(PassManager &pm) {
    // Storage-level aliasing must be resolved before anything reasons about
    // which buffer a node touches; cheap and idempotent after the first call.
    //
    // Recursively, at every depth: passes that rewrite loop bodies ask
    // ``Graph::resolve_alias`` which buffer a node touches, and an unlinked
    // body answers "aliases nothing", so Reorder could move a writer past a
    // reader of the same buffer.
    link_alias_storage();
    for_each_descendant([](Graph &sub) { sub.link_alias_storage(); });
    std::scoped_lock const lock(_content_mutex);
    bool const             modified = pm.run(*this);
    if (modified) {
        note(Change::PassesApplied);
    }
    return modified;
}

bool Graph::optimize() {
    return optimize(OptLevel::O2);
}

bool Graph::optimize(OptLevel level) {
    size_t const before = _nodes.size();

    auto       pm       = PassManager::create_for(level);
    bool const modified = apply(pm);

    _last_optimize_report =
        fmt::format("optimize(O{}) on '{}': {} -> {} node(s)\n{}", static_cast<int>(level), _name, before, _nodes.size(), pm.explain());
    _last_decisions = pm.decisions();
    return modified;
}

EINSUMS_NAMESPACE_END(compute_graph)
