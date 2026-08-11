//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Executor.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// Execute a single node with timing, recording results in the graph.
void execute_node(Node &node) {
    if (node.execute) {
        // Prefer the synchronous executor when available.
        node.execute();
    } else if (node.async_start && node.async_finish) {
        // Fallback: run async phases synchronously (start + wait).
        // The DataflowExecutor handles true overlap; Sequential/MPI just runs serially.
        node.async_start();
        node.async_finish();
    }
}

/// Run every node in list order, collecting timings into one batch.
///
/// The batch matters: recording per node took the graph's RECURSIVE content
/// mutex and copied the node's label string, once per node per replay, which
/// an SCF/CC loop pays on every iteration. Samples carry only the node id, and
/// Graph::timing_report() resolves labels if anyone asks for them.
void execute_all_timed(Graph &graph) {
    auto &nodes = graph.nodes();

    std::vector<Graph::NodeTimingSample> samples;
    samples.reserve(nodes.size());
    for (auto &node : nodes) {
        auto t0 = std::chrono::steady_clock::now();
        execute_node(node);
        auto t1 = std::chrono::steady_clock::now();

        samples.push_back({.id = node.id, .kind = node.kind, .duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count()});
    }
    graph.record_node_timings(std::move(samples));
}

} // namespace

// ─── SequentialExecutor ─────────────────────────────────────────────────────

void SequentialExecutor::execute(Graph &graph) {
    execute_all_timed(graph);
}

// ─── OpenMPExecutor ─────────────────────────────────────────────────────────

void OpenMPExecutor::execute(Graph &graph) {
    auto        &nodes = graph.nodes();
    auto const  &deps  = graph.dependencies();
    size_t const n     = nodes.size();

    if (n == 0)
        return;

#ifdef _OPENMP
        // Everything in a level is launched together, so a conflict inside one is
        // a data race rather than a slow schedule - and one that leaves every
        // serial replay correct, so nothing else will report it. On by default
        // where asserts are, and available in a release build behind
        // ``--einsums:graph:verify-levels`` for chasing a result that moves
        // between runs.
        // Only the DEFAULT is conditional, so the call below compiles in every
        // configuration; a debug-only block would be built by nobody who runs the
        // usual RelWithDebInfo tree and would rot unnoticed.
#    if defined(EINSUMS_DEBUG)
    constexpr bool verify_by_default = true;
#    else
    constexpr bool verify_by_default = false;
#    endif
    if (GlobalConfigMap::get_singleton().get_bool("graph-verify-levels", verify_by_default)) {
        graph.verify_level_independence();
    }

    // Level partition comes precomputed with the dependency lists (it used
    // to be re-derived here on every execute).
    auto const &levels = deps.levels;

    // Lock-free timing: every node writes its own slot (distinct indices,
    // no sharing), merged serially below in deterministic node order. The
    // old path serialized every node on a shared mutex.
    std::vector<double> node_ms(n, -1.0);

    // An exception must NOT escape an OpenMP structured block, doing so leaves
    // the team waiting at the join barrier forever (deadlock). Catch any node
    // failure inside the region, keep the first, and rethrow after the barrier.
    std::exception_ptr first_exc;
    std::mutex         exc_mutex;

    for (auto const &group : levels) {
        if (group.size() == 1) {
            size_t const i  = group[0];
            auto         t0 = std::chrono::steady_clock::now();
            execute_node(nodes[i]);
            auto t1    = std::chrono::steady_clock::now();
            node_ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        } else {
#    pragma omp parallel for schedule(dynamic)
            for (size_t g = 0; g < group.size(); g++) { // NOLINT(modernize-loop-convert)
                size_t const i = group[g];
                try {
                    auto t0 = std::chrono::steady_clock::now();
                    execute_node(nodes[i]);
                    auto t1    = std::chrono::steady_clock::now();
                    node_ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                } catch (...) {
                    std::scoped_lock const lock(exc_mutex);
                    if (!first_exc) {
                        first_exc = std::current_exception();
                    }
                }
            }
            if (first_exc) {
                std::rethrow_exception(first_exc); // safely outside the parallel region
            }
        }
    }

    std::vector<Graph::NodeTimingSample> samples;
    samples.reserve(n);
    for (size_t i = 0; i < n; i++) {
        if (node_ms[i] >= 0.0) {
            samples.push_back({.id = nodes[i].id, .kind = nodes[i].kind, .duration_ms = node_ms[i]});
        }
    }
    graph.record_node_timings(std::move(samples));
#else
    execute_all_timed(graph);
#endif
}

// ─── DataflowExecutor ──────────────────────────────────────────────────────

/// Per-run scheduling state, pooled and reused across replays.
///
/// Everything here is either reset per run (the counters, the timing slots) or
/// reused verbatim (the allocations behind them, the interned zone names). The
/// graph pointers are refreshed at the top of every run, so a scaffold never
/// outlives its knowledge of a graph: it is a bag of buffers, not a cache of
/// graph structure, which is what makes reuse safe without an identity check.
struct DataflowExecutor::Scaffold {
    /// One node's wall-clock time, padded to its own cache line.
    ///
    /// A contiguous vector<double> put eight nodes' slots on one line, so every
    /// worker writing a completed node's time invalidated the line under seven
    /// others - false sharing on a value nobody reads until the run is over.
    struct alignas(64) PaddedMs {
        double ms{-1.0};
    };

    // ── Refreshed per run ───────────────────────────────────────────────────
    std::vector<Node>    *nodes{nullptr};
    DependencyInfo const *deps{nullptr};
    task_pool::TaskPool  *pool{nullptr};
    size_t                n{0};
    size_t                budget{0};
    bool                  recording{false}; ///< profiler state, read once per run

    // ── Reset per run, allocation reused ────────────────────────────────────
    std::vector<std::atomic<int>> remaining; ///< preds left per node
    std::atomic<size_t>           completed{0};
    std::atomic<bool>             failed{false};
    std::exception_ptr            first_exc;
    std::mutex                    exc_mutex;

    /// Per-node timing slots: each task writes only its own index, so no lock
    /// is needed. Slot writes happen-before the task's completed increment
    /// (release), and the merge runs after help_until observes completed == n
    /// (acquire).
    std::vector<PaddedMs> node_ms;

    // Memory budget: Materialize nodes that would exceed the budget wait in
    // `deferred` (instead of blocking a worker) and are resubmitted when a
    // Free node returns bytes.
    std::atomic<size_t> mem_current{0};
    std::mutex          deferred_mutex;
    std::vector<size_t> deferred;

    /// Scratch for the root-seeding batch, reused across replays.
    std::vector<std::function<void()>> root_batch;

    /// Interned zone name per node, so a profiled run does not intern a label
    /// per task. Only filled while recording; keyed to the graph and node list
    /// it was built for, since a wrong name is cosmetic but a stale one is
    /// confusing.
    std::vector<uint32_t> zone_ids;
    Graph const          *zone_graph{nullptr};
    std::uint64_t         zone_version{0};

    /// Size the reused buffers for an n-node run and clear the per-run state.
    void reset(Graph &graph, size_t node_count);

    /// Rebuild zone_ids if this graph's labels are not the ones cached.
    void refresh_zone_ids(Graph const &graph);

    /// Submit node @p i. With @p batch non-null the closure is appended there
    /// instead of enqueued, for the caller to hand to the pool in one go.
    void submit(size_t i, std::vector<std::function<void()>> *batch = nullptr);
    void complete(size_t i);
    void drain_deferred();
    void run_node(size_t i);
    void run_async_start(size_t i);
    void run_async_finish(size_t i);

    /// Record @p e as the run's failure if it is the first one.
    void fail(std::exception_ptr e) {
        std::scoped_lock const lock(exc_mutex);
        if (!first_exc) {
            first_exc = std::move(e);
        }
        failed.store(true, std::memory_order_release);
    }
};

namespace {

/// The one call site every dataflow task's zone reports; the NAME comes from
/// the node and is interned once per graph in Scaffold::zone_ids.
profile::ZoneSite const &dataflow_task_site() {
    static profile::ZoneSite const site{"dataflow task", __FILE__, __LINE__, __func__};
    return site;
}

} // namespace

void DataflowExecutor::Scaffold::reset(Graph &graph, size_t node_count) {
    nodes = &graph.nodes();
    deps  = &graph.dependencies();
    pool  = &task_pool::TaskPool::get_singleton();
    n     = node_count;

    // atomic<int> is neither movable nor copyable, so the counter vector can
    // only be resized by rebuilding it; same-size replays skip that.
    if (remaining.size() != n) {
        remaining = std::vector<std::atomic<int>>(n);
    }
    for (size_t i = 0; i < n; i++) {
        remaining[i].store(static_cast<int>(deps->predecessors[i].size()), std::memory_order_relaxed);
    }

    node_ms.assign(n, PaddedMs{});
    completed.store(0, std::memory_order_relaxed);
    failed.store(false, std::memory_order_relaxed);
    first_exc = nullptr;
    mem_current.store(0, std::memory_order_relaxed);
    deferred.clear();
}

void DataflowExecutor::Scaffold::refresh_zone_ids(Graph const &graph) {
    if (zone_graph == &graph && zone_version == graph.analysis_version() && zone_ids.size() == n) {
        return;
    }
    zone_ids.resize(n);
    for (size_t i = 0; i < n; i++) {
        zone_ids[i] = profile::intern_string((*nodes)[i].label);
    }
    zone_graph   = &graph;
    zone_version = graph.analysis_version();
}

void DataflowExecutor::Scaffold::complete(size_t i) {
    // A successor submission that throws (an allocation failure in the pool,
    // say) must not skip the counter: help_until() waits for completed == n and
    // would never return.
    try {
        for (size_t const succ : (*deps).successors[i]) {
            if (remaining[succ].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                submit(succ);
            }
        }
    } catch (...) {
        fail(std::current_exception());
    }
    completed.fetch_add(1, std::memory_order_release);
}

void DataflowExecutor::Scaffold::drain_deferred() {
    // Called after a Free returns bytes: resubmit deferred Materialize nodes
    // that now fit, charging them under the lock so concurrent drains don't
    // double-book the budget.
    std::vector<size_t> runnable;
    {
        std::scoped_lock const lk(deferred_mutex);
        for (auto it = deferred.begin(); it != deferred.end();) {
            size_t const bytes = (*nodes)[*it].estimated_bytes;
            if (mem_current.load(std::memory_order_relaxed) + bytes <= budget) {
                mem_current.fetch_add(bytes, std::memory_order_relaxed);
                runnable.push_back(*it);
                it = deferred.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (size_t const i : runnable) {
        submit(i);
    }
}

void DataflowExecutor::Scaffold::run_node(size_t i) {
    Node &node = (*nodes)[i];

    bool const   is_free    = budget > 0 && node.kind == OpKind::Free;
    size_t const free_bytes = is_free ? node.estimated_bytes : 0;

    // After any failure the remaining nodes are drained without executing
    // (execute() rethrows the first exception; partial results are unspecified
    // either way).
    if (!failed.load(std::memory_order_acquire)) {
        std::optional<profile::ScopedZone> zone;
        if (recording) {
            zone.emplace(dataflow_task_site(), zone_ids[i], node.label);
        }
        try {
            auto t0 = std::chrono::steady_clock::now();
            execute_node(node);
            auto t1 = std::chrono::steady_clock::now();

            node_ms[i].ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        } catch (...) {
            fail(std::current_exception());
        }
    }

    if (is_free && free_bytes > 0) {
        mem_current.fetch_sub(free_bytes, std::memory_order_relaxed);
        drain_deferred();
    }
    complete(i);
}

void DataflowExecutor::Scaffold::run_async_start(size_t i) {
    Node &node = (*nodes)[i];
    if (!failed.load(std::memory_order_acquire)) {
        std::optional<profile::ScopedZone> zone;
        if (recording) {
            zone.emplace(dataflow_task_site(), zone_ids[i], node.label);
        }
        try {
            node.async_start();
        } catch (...) {
            fail(std::current_exception());
        }
    }
    // The finish half is a second task so other ready nodes can interleave
    // between them, which is the whole point of splitting an async node.
    try {
        pool->submit_bare([this, i]() { run_async_finish(i); });
    } catch (...) {
        fail(std::current_exception());
        complete(i);
    }
}

void DataflowExecutor::Scaffold::run_async_finish(size_t i) {
    Node &node = (*nodes)[i];
    if (!failed.load(std::memory_order_acquire)) {
        std::optional<profile::ScopedZone> zone;
        if (recording) {
            zone.emplace(dataflow_task_site(), zone_ids[i], node.label);
        }
        try {
            auto t0 = std::chrono::steady_clock::now();
            node.async_finish();
            auto t1 = std::chrono::steady_clock::now();

            node_ms[i].ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        } catch (...) {
            fail(std::current_exception());
        }
    }
    complete(i);
}

void DataflowExecutor::Scaffold::submit(size_t i, std::vector<std::function<void()>> *batch) {
    Node &node = (*nodes)[i];

    // Budget gate: park over-budget Materialize nodes instead of submitting
    // them (never blocks a worker; Frees are never gated, so parked
    // allocations always drain).
    if (budget > 0 && node.kind == OpKind::Materialize && node.estimated_bytes > 0) {
        // A single materialization larger than the WHOLE budget can never be
        // scheduled (drain_deferred only re-runs a node that fits). Parking it
        // left it in the deferred queue forever, so completed never reached n
        // and help_until() deadlocked the calling thread. Fail the run instead.
        if (node.estimated_bytes > budget) {
            fail(std::make_exception_ptr(
                std::runtime_error("DataflowExecutor: node '" + node.label + "' requires more memory than the configured budget")));
            complete(i);
            return;
        }
        std::scoped_lock const lk(deferred_mutex);
        if (mem_current.load(std::memory_order_relaxed) + node.estimated_bytes > budget) {
            deferred.push_back(i);
            return;
        }
        mem_current.fetch_add(node.estimated_bytes, std::memory_order_relaxed);
    }

    // Both closures capture exactly a pointer and an index, which fits inside
    // std::function's inline buffer - so submitting a node allocates nothing.
    // The old path went through submit_detached(node.label, lambda), which
    // copied the label into a wrapper closure and heap-allocated the wrapper
    // AND the wrapped callable, per node per replay. The zone that wrapper
    // provided is now taken inside the run_* bodies from a pre-interned name.
    //
    // `batch` is passed only by the root-seeding loop, which runs on the
    // calling thread and so would otherwise take the pool's external-queue
    // mutex and signal its condition variable once per root. Successors are
    // submitted from worker threads (own-deque push, no lock) and from
    // drain_deferred, which can run concurrently with the seeding loop - hence
    // an explicit parameter rather than scaffold state those paths could race
    // on.
    if (node.async_start && node.async_finish) {
        if (batch != nullptr) {
            batch->emplace_back([this, i]() { run_async_start(i); });
        } else {
            pool->submit_bare([this, i]() { run_async_start(i); });
        }
    } else {
        if (batch != nullptr) {
            batch->emplace_back([this, i]() { run_node(i); });
        } else {
            pool->submit_bare([this, i]() { run_node(i); });
        }
    }
}

DataflowExecutor::DataflowExecutor()  = default;
DataflowExecutor::~DataflowExecutor() = default;

void DataflowExecutor::execute(Graph &graph) {
    size_t const n = graph.nodes().size();

    if (n == 0)
        return;

    // Counter-based dataflow scheduling. Only the dependency-free roots are
    // submitted from this thread; every other node is submitted by the worker
    // that completed its last predecessor (an own-deque push, no lock), and
    // readiness is tracked with plain atomic countdowns taken from the graph's
    // cached dependency lists.
    //
    // Take a scaffold for the duration of the run so nested and concurrent
    // runs never share one (see the pool's comment in Executor.hpp).
    std::unique_ptr<Scaffold> scaffold;
    {
        std::scoped_lock const lk(_scaffold_mutex);
        if (!_scaffolds.empty()) {
            scaffold = std::move(_scaffolds.back());
            _scaffolds.pop_back();
        }
    }
    if (!scaffold) {
        scaffold = std::make_unique<Scaffold>();
    }

    Scaffold &s = *scaffold;
    s.reset(graph, n);
    s.budget    = _memory_budget;
    s.recording = profile::Profiler::instance().enabled();
    if (s.recording) {
        s.refresh_zone_ids(graph);
    }

    auto const &deps = *s.deps;
    auto       &pool = *s.pool;

    // Seed the dependency-free roots in one batch; everything else
    // self-schedules. Submitting them one at a time cost a lock and a
    // condition-variable signal each, which on a wide graph was most of the
    // scheduling time.
    s.root_batch.clear();
    for (size_t i = 0; i < n; i++) {
        if (deps.predecessors[i].empty()) {
            s.submit(i, &s.root_batch);
        }
    }
    pool.submit_bare_batch(s.root_batch);

    // The calling thread work-steals while waiting instead of parking.
    pool.help_until([&s]() { return s.completed.load(std::memory_order_acquire) == s.n; });

    // Serial merge in deterministic node order (the mutex-per-node this
    // replaces produced completion order, which was nondeterministic anyway).
    auto                                &nodes = graph.nodes();
    std::vector<Graph::NodeTimingSample> samples;
    samples.reserve(n);
    for (size_t i = 0; i < n; i++) {
        if (s.node_ms[i].ms >= 0.0) {
            samples.push_back({.id = nodes[i].id, .kind = nodes[i].kind, .duration_ms = s.node_ms[i].ms});
        }
    }
    graph.record_node_timings(std::move(samples));

    std::exception_ptr const exc = s.first_exc;

    // Return the scaffold whether or not the run failed; every task has
    // completed by now (help_until only returns at completed == n), so nothing
    // still refers to it.
    {
        std::scoped_lock const lk(_scaffold_mutex);
        _scaffolds.push_back(std::move(scaffold));
    }

    if (exc) {
        std::rethrow_exception(exc);
    }
}

// ─── MPIExecutor ────────────────────────────────────────────────────────────

void MPIExecutor::execute(Graph &graph) {
    // All ranks execute the same node sequence. Compute nodes operate on
    // local partitions; communication nodes are collective (all ranks
    // participate). On mock backend (1 rank), this is identical to Sequential.
    execute_all_timed(graph);
}

EINSUMS_NAMESPACE_END(compute_graph)
