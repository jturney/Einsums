//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraph/Executor.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>
#include <Einsums/TaskPool/WidthBudget.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
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

/// Gives the calling thread @p width threads for as long as it lives, and hands
/// back exactly what it found.
///
/// Restoring the SAVED values rather than the literal 1 is the whole point.
/// Pool workers pin themselves to one thread at startup and every unplanned
/// task depends on that pin, so restoring 1 is right for them - but
/// `help_until` also runs tasks on the thread that called `execute()`, which is
/// not pinned, and leaving that thread narrowed would follow the caller into
/// whatever it does after the replay returns.
///
/// Only constructed for widths above 1, so an unplanned graph reads no ICV and
/// writes none.
class WidthGuard {
  public:
    explicit WidthGuard(int width) {
#ifdef _OPENMP
        _prev_omp = omp_get_max_threads();
        omp_set_num_threads(width);
#endif
        // The raised ICV is for the kernels einsums threads itself; an
        // OpenMP-threaded vendor BLAS must not fork from it (concurrent
        // callers with differing ICVs are outside what OpenBLAS supports, see
        // blas::set_moldable_width_scope). The flag makes the BLAS wrappers
        // clamp any vendor call made under this guard back to one thread.
        _prev_scope = blas::moldable_width_scope();
        blas::set_moldable_width_scope(true);
        // A vendor that cannot be read cannot be set either (both are the same
        // build-time switch), so a zero here means there is no vendor state to
        // save, and restoring the 0 would ask for a nonsense thread count.
        _prev_blas = blas::get_num_threads_this_thread();
        if (_prev_blas > 0) {
            blas::set_num_threads_this_thread(width);
        }
    }

    ~WidthGuard() {
        if (_prev_blas > 0) {
            blas::set_num_threads_this_thread(_prev_blas);
        }
        blas::set_moldable_width_scope(_prev_scope);
#ifdef _OPENMP
        omp_set_num_threads(_prev_omp);
#endif
    }

    WidthGuard(WidthGuard const &)            = delete;
    WidthGuard &operator=(WidthGuard const &) = delete;
    WidthGuard(WidthGuard &&)                 = delete;
    WidthGuard &operator=(WidthGuard &&)      = delete;

  private:
#ifdef _OPENMP
    int _prev_omp{1};
#endif
    int  _prev_blas{0};
    bool _prev_scope{false};
};

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

/// Replays that ran width-1 because the plan was made for another machine.
std::atomic<std::size_t> g_stale_thread_plan_fallbacks{0};

} // namespace

std::size_t stale_thread_plan_fallbacks() {
    return g_stale_thread_plan_fallbacks.load(std::memory_order_relaxed);
}

void reset_stale_thread_plan_fallbacks() {
    g_stale_thread_plan_fallbacks.store(0, std::memory_order_relaxed);
}

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
    // The build-dependent default is the descriptor's, so there is no
    // second one here to drift out of step with it.
    if (config::get(option::GraphVerifyLevels)) {
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

    // ── Width budget ────────────────────────────────────────────────────────
    //
    // Off unless some node of this graph carries a width above 1, and then off
    // again if that plan was made for a different machine. An unplanned graph -
    // which is every graph until a planning pass runs - takes not one atomic
    // more than it did before widths existed, which is what keeps the ~0.44 us
    // per-node submission floor where it is.
    bool widths_active{false};

    /// Admission order, computed once per run: the longest remaining path from
    /// a node to a sink, so a node on a long tail is admitted before a node
    /// that finishes the graph.
    ///
    /// A planned graph carries the path in ESTIMATED TIME under its chosen
    /// widths (@ref Node::admission_priority, written by ThreadPlanning), which
    /// is the same ordering with real numbers in it - a long tail of tiny nodes
    /// should not outrank one fat node. A graph nobody planned falls back to
    /// the structural hop count (@ref fill_structural_priorities), which needs
    /// no cost model and is the ordering the executor had before widths.
    std::vector<std::int64_t> priority;

    /// Width charged to the budget per node, 0 for a node that never acquired
    /// one. Written by the thread that admits the node, before it queues the
    /// task, and read by the thread that completes it.
    std::vector<std::atomic<unsigned>> admitted;

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

    /// Fill @ref priority from the plan the nodes carry, or from the structural
    /// hop count when they carry none.
    void fill_priorities();

    /// Fill @ref priority with each node's longest hop count to a sink.
    void fill_structural_priorities();

    /// The width node @p i must be admitted for, 0 when nothing is gated.
    [[nodiscard]] unsigned effective_width(Node const &node) const;

    /// Submit node @p i. With @p batch non-null the closure is appended there
    /// instead of enqueued, for the caller to hand to the pool in one go.
    void submit(size_t i, std::vector<std::function<void()>> *batch = nullptr);

    /// Second half of submit, for a node whose memory is already charged:
    /// acquire its width, then queue it. Re-entering submit() here instead
    /// would charge the bytes a second time.
    void admit(size_t i, std::vector<std::function<void()>> *batch = nullptr);

    /// Third half: hand the node's task to the pool (or to @p batch).
    void enqueue_task(size_t i, std::vector<std::function<void()>> *batch);

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
    // The width scan rides along with the counter fill rather than walking the
    // node list a second time.
    bool any_wide = false;
    for (size_t i = 0; i < n; i++) {
        remaining[i].store(static_cast<int>(deps->predecessors[i].size()), std::memory_order_relaxed);
        any_wide = any_wide || (*nodes)[i].thread_width > 1;
    }

    node_ms.assign(n, PaddedMs{});
    completed.store(0, std::memory_order_relaxed);
    failed.store(false, std::memory_order_relaxed);
    first_exc = nullptr;
    mem_current.store(0, std::memory_order_relaxed);
    deferred.clear();

    widths_active = any_wide;
    if (any_wide) {
        auto &width_budget = task_pool::WidthBudget::get_singleton();
        // Asked before anything is admitted and from the thread that starts the
        // run, which is the only one that can see the machine (see
        // WidthBudget::sync_machine_width).
        width_budget.sync_machine_width();

        // Plan staleness. A width is a share of a known number of threads, so a
        // plan for a different number is not conservative, it is wrong: it can
        // hand out more of the machine than exists, or leave most of it idle.
        // Falling back to width 1 is exactly today's behavior for an unplanned
        // graph, which is the one plan that is right on every machine.
        unsigned const planned = graph.planned_thread_count();
        if (planned != 0 && planned != width_budget.total()) {
            widths_active = false;
            g_stale_thread_plan_fallbacks.fetch_add(1, std::memory_order_relaxed);
            static std::once_flag warned;
            std::call_once(warned, [&]() {
                EINSUMS_LOG_WARN("ComputeGraph: graph '{}' carries thread widths planned for {} threads but the machine is rationing {}; "
                                 "running every node at width 1. Re-plan to use the machine.",
                                 graph.name(), planned, width_budget.total());
            });
        }
    }

    if (widths_active) {
        if (admitted.size() != n) {
            admitted = std::vector<std::atomic<unsigned>>(n);
        }
        for (size_t i = 0; i < n; i++) {
            admitted[i].store(0, std::memory_order_relaxed);
        }
        fill_priorities();
    }
}

void DataflowExecutor::Scaffold::fill_priorities() {
    // A planner writes a nonzero priority on every node it plans, so one
    // nonzero anywhere means these numbers are the plan's and not left over
    // from a hand-set width. Estimated nanoseconds to a sink, so they order
    // exactly as the hop counts do but with the node times in them.
    for (size_t i = 0; i < n; i++) {
        if ((*nodes)[i].admission_priority != 0) {
            priority.assign(n, 0);
            for (size_t k = 0; k < n; k++) {
                priority[k] = (*nodes)[k].admission_priority;
            }
            return;
        }
    }
    fill_structural_priorities();
}

void DataflowExecutor::Scaffold::fill_structural_priorities() {
    // Node positions are topological (DependencyInfo's contract), so every
    // successor of i sits at a position above i and one backward pass gives
    // each node the longest hop count to a sink.
    priority.assign(n, 0);
    for (size_t k = n; k-- > 0;) {
        std::int64_t longest = 0;
        for (size_t const succ : deps->successors[k]) {
            longest = std::max(longest, priority[succ]);
        }
        priority[k] = longest + 1;
    }
}

unsigned DataflowExecutor::Scaffold::effective_width(Node const &node) const {
    if (!widths_active) {
        return 0;
    }
    // A control-flow node holds one unit while its body replays, so the machine
    // is never handed out twice; the body's own nodes acquire their real widths
    // from the same budget, and the unit this one holds goes back to the budget
    // for the duration of the nested run (WidthBudget::BlockedScope).
    if (node.kind == OpKind::Loop || node.kind == OpKind::Conditional) {
        return 1;
    }
    return node.thread_width == 0 ? 1U : node.thread_width;
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
    // Width goes back BEFORE the successors are submitted, so a successor can
    // be admitted into the room this node just left instead of parking behind
    // it. Nothing about the release depends on the node having run: a node that
    // threw, or that was drained after a failure, returns its width here too,
    // which is what keeps a failed run from wedging every later one.
    if (widths_active) {
        unsigned const width = admitted[i].exchange(0, std::memory_order_relaxed);
        if (width > 0) {
            // Releasing runs the parked tasks it admits, so it inherits their
            // failure modes; the counter below must be reached whatever they do
            // with them, for the same reason the successor loop is guarded.
            try {
                task_pool::WidthBudget::get_singleton().release(width);
            } catch (...) {
                fail(std::current_exception());
            }
        }
    }

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
    // admit(), not submit(): the bytes were charged above, and going back
    // through the memory gate would charge them a second time.
    for (size_t const i : runnable) {
        admit(i);
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
            // Tells this thread what it holds, so a nested replay started from
            // inside the kernel can lend the width back while it waits. Costs
            // nothing for a node that acquired nothing.
            task_pool::WidthBudget::HoldScope const hold(widths_active ? admitted[i].load(std::memory_order_relaxed) : 0U);
            // A planned width is honored by giving the executing thread that
            // many threads for the node and no longer. Widths of 0 (unplanned)
            // and 1 take the original path untouched: no thread-count is read,
            // none is written, and the per-node submission cost is unchanged.
            // A stale plan reaches here with widths_active false and is run as
            // though every node were unplanned.
            if (widths_active && node.thread_width > 1) {
                WidthGuard const width(node.thread_width);
                execute_node(node);
            } else {
                execute_node(node);
            }
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

/// The async halves deliberately do NOT honor Node::thread_width. The two run as
/// separate tasks and so on two arbitrary threads, and the width contract is a
/// property of one thread over one interval - there is no interval here that
/// covers the operation, which by construction proceeds outside both tasks.
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

    admit(i, batch);
}

void DataflowExecutor::Scaffold::admit(size_t i, std::vector<std::function<void()>> *batch) {
    // The whole width machinery is skipped for a graph nobody planned, down to
    // the branch that reads the node: no budget lock, no atomic, no priority
    // lookup on the path every replay takes today.
    if (!widths_active) {
        enqueue_task(i, batch);
        return;
    }

    // The deferred_mutex is NOT held here (submit released it, drain_deferred
    // collects under it and calls in after releasing), and the budget invokes
    // continuations with its own lock released. Neither lock is ever taken
    // inside the other, in either order.
    unsigned const granted = task_pool::WidthBudget::get_singleton().acquire(
        effective_width((*nodes)[i]), {.rank = priority[i], .tiebreak = i}, [this, i](unsigned width) {
            admitted[i].store(width, std::memory_order_relaxed);
            try {
                enqueue_task(i, nullptr);
            } catch (...) {
                // The width is already charged and nobody else will ever
                // complete this node, so failing to queue it would both hang
                // the run and lose the units to the process for good. Complete
                // it here instead: the width goes back and the count advances.
                fail(std::current_exception());
                complete(i);
            }
        });

    if (granted == 0) {
        // Parked. The continuation owns the node now, and may in fact have run
        // already, so nothing more may be done with it here.
        return;
    }
    admitted[i].store(granted, std::memory_order_relaxed);
    enqueue_task(i, batch);
}

void DataflowExecutor::Scaffold::enqueue_task(size_t i, std::vector<std::function<void()>> *batch) {
    Node &node = (*nodes)[i];

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

    // If this run was started from inside an admitted task - a Loop or
    // Conditional node replaying its body - that task is about to wait for the
    // whole of this run and is not computing while it does. Its width goes back
    // to the budget for the duration, and is taken again when this returns.
    // Without that, a body node planned at the machine's full width could never
    // be admitted (an ancestor holds a unit of it) and the nested run would
    // never finish.
    task_pool::WidthBudget::BlockedScope const lend_width_to_body;

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
