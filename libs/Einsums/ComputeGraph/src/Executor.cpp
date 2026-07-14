//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Executor.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace einsums::compute_graph {

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

void execute_timed(Node &node, Graph &graph) {
    auto t0 = std::chrono::steady_clock::now();
    execute_node(node);
    auto t1 = std::chrono::steady_clock::now();

    double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    graph.record_node_timing(node.id, node.label, node.kind, ms);
}

/// Thread-safe version for parallel executors.
void execute_timed_mt(Node &node, Graph &graph, std::mutex &timing_mutex) {
    auto t0 = std::chrono::steady_clock::now();
    execute_node(node);
    auto t1 = std::chrono::steady_clock::now();

    double const           ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::scoped_lock const lock(timing_mutex);
    graph.record_node_timing(node.id, node.label, node.kind, ms);
}

} // namespace

// ─── SequentialExecutor ─────────────────────────────────────────────────────

void SequentialExecutor::execute(Graph &graph) {
    for (auto &node : graph.nodes()) {
        execute_timed(node, graph);
    }
}

// ─── OpenMPExecutor ─────────────────────────────────────────────────────────

void OpenMPExecutor::execute(Graph &graph) {
    auto        &nodes = graph.nodes();
    auto const  &deps  = graph.dependencies();
    size_t const n     = nodes.size();

    if (n == 0)
        return;

    std::mutex timing_mutex;

#ifdef _OPENMP
    std::vector<size_t> level(n, 0);
    for (size_t i = 0; i < n; i++) {
        for (size_t const pred : deps.predecessors[i]) {
            if (level[pred] + 1 > level[i]) {
                level[i] = level[pred] + 1;
            }
        }
    }

    size_t max_level = 0;
    for (size_t i = 0; i < n; i++) {
        if (level[i] > max_level)
            max_level = level[i];
    }

    std::vector<std::vector<size_t>> levels(max_level + 1);
    for (size_t i = 0; i < n; i++) {
        levels[level[i]].push_back(i);
    }

    // An exception must NOT escape an OpenMP structured block, doing so leaves
    // the team waiting at the join barrier forever (deadlock). Catch any node
    // failure inside the region, keep the first, and rethrow after the barrier.
    std::exception_ptr first_exc;
    std::mutex         exc_mutex;

    for (auto const &group : levels) {
        if (group.size() == 1) {
            execute_timed(nodes[group[0]], graph);
        } else {
#    pragma omp parallel for schedule(dynamic)
            for (size_t g = 0; g < group.size(); g++) { // NOLINT(modernize-loop-convert)
                try {
                    execute_timed_mt(nodes[group[g]], graph, timing_mutex);
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
#else
    for (auto &node : nodes) {
        execute_timed(node, graph);
    }
#endif
}

// ─── DataflowExecutor ──────────────────────────────────────────────────────

void DataflowExecutor::execute(Graph &graph) {
    auto        &nodes = graph.nodes();
    auto const  &deps  = graph.dependencies();
    size_t const n     = nodes.size();

    if (n == 0)
        return;

    auto &pool = task_pool::TaskPool::get_singleton();

    // Counter-based dataflow scheduling. The old implementation rebuilt a
    // full TaskHandle/when_all scaffold per execute (per-node handle vectors,
    // combined shared states, "/start"+"/finish" label concatenations) and
    // submitted every node from this thread - each submission taking the
    // pool's external mutex and waking every worker. Here only the
    // dependency-free roots are submitted externally; every other node is
    // submitted by the worker that completed its last predecessor (an
    // own-deque push, no lock), and readiness is tracked with plain atomic
    // countdowns taken from the graph's cached dependency lists.
    struct RunState {
        std::vector<std::atomic<int>> remaining; ///< preds left per node
        std::atomic<size_t>           completed{0};
        std::atomic<bool>             failed{false};
        std::exception_ptr            first_exc;
        std::mutex                    exc_mutex;
        std::mutex                    timing_mutex;

        // Memory budget: Materialize nodes that would exceed the budget wait
        // in `deferred` (instead of blocking a worker) and are resubmitted
        // when a Free node returns bytes.
        size_t              budget{0};
        std::atomic<size_t> mem_current{0};
        std::mutex          deferred_mutex;
        std::vector<size_t> deferred;
    };
    auto state       = std::make_shared<RunState>();
    state->budget    = _memory_budget;
    state->remaining = std::vector<std::atomic<int>>(n);
    for (size_t i = 0; i < n; i++) {
        state->remaining[i].store(static_cast<int>(deps.predecessors[i].size()), std::memory_order_relaxed);
    }

    // The scheduling callables live on this frame: help_until() below does
    // not return until every task has completed, and no task touches them
    // after its final completed-counter increment, so reference captures
    // into task bodies are safe and avoid shared_ptr cycles.
    std::function<void(size_t)> submit_node;

    auto record_failure = [state]() {
        std::scoped_lock const lock(state->exc_mutex);
        if (!state->first_exc) {
            state->first_exc = std::current_exception();
        }
        state->failed.store(true, std::memory_order_release);
    };

    auto complete_node = [state, &deps, &submit_node](size_t i) {
        for (size_t const succ : deps.successors[i]) {
            if (state->remaining[succ].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                submit_node(succ);
            }
        }
        state->completed.fetch_add(1, std::memory_order_release);
    };

    auto drain_deferred = [state, &submit_node, &nodes]() {
        // Called after a Free returns bytes: resubmit deferred Materialize
        // nodes that now fit, charging them under the lock so concurrent
        // drains don't double-book the budget.
        std::vector<size_t> runnable;
        {
            std::scoped_lock const lk(state->deferred_mutex);
            for (auto it = state->deferred.begin(); it != state->deferred.end();) {
                size_t const bytes = nodes[*it].estimated_bytes;
                if (state->mem_current.load(std::memory_order_relaxed) + bytes <= state->budget) {
                    state->mem_current.fetch_add(bytes, std::memory_order_relaxed);
                    runnable.push_back(*it);
                    it = state->deferred.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (size_t const i : runnable) {
            submit_node(i);
        }
    };

    auto run_body = [state, &graph](Node &node, auto &&body) {
        // After any failure the remaining nodes are drained without
        // executing (execute() rethrows the first exception; partial
        // results are unspecified either way).
        if (state->failed.load(std::memory_order_acquire)) {
            return;
        }
        try {
            auto t0 = std::chrono::steady_clock::now();
            body();
            auto         t1 = std::chrono::steady_clock::now();
            double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::scoped_lock const lock(state->timing_mutex);
            graph.record_node_timing(node.id, node.label, node.kind, ms);
        } catch (...) {
            std::scoped_lock const lock(state->exc_mutex);
            if (!state->first_exc) {
                state->first_exc = std::current_exception();
            }
            state->failed.store(true, std::memory_order_release);
        }
    };

    submit_node = [state, &submit_node, &complete_node, &drain_deferred, &run_body, &record_failure, &pool, &nodes](size_t i) {
        Node &node = nodes[i];

        // Budget gate: park over-budget Materialize nodes instead of
        // submitting them (never blocks a worker; Frees are never gated,
        // so parked allocations always drain).
        if (state->budget > 0 && node.kind == OpKind::Materialize && node.estimated_bytes > 0) {
            std::scoped_lock const lk(state->deferred_mutex);
            if (state->mem_current.load(std::memory_order_relaxed) + node.estimated_bytes > state->budget) {
                state->deferred.push_back(i);
                return;
            }
            state->mem_current.fetch_add(node.estimated_bytes, std::memory_order_relaxed);
        }

        bool const has_async = static_cast<bool>(node.async_start) && static_cast<bool>(node.async_finish);
        if (has_async) {
            // Two chained tasks: start when ready, finish afterwards (other
            // ready nodes can interleave between them for real overlap).
            pool.submit_detached(node.label, [state, &complete_node, &run_body, &record_failure, &pool, &node, i]() {
                if (!state->failed.load(std::memory_order_acquire)) {
                    try {
                        node.async_start();
                    } catch (...) {
                        record_failure();
                    }
                }
                pool.submit_detached(node.label, [&complete_node, &run_body, &node, i]() {
                    run_body(node, [&node]() { node.async_finish(); });
                    complete_node(i);
                });
            });
            return;
        }

        bool const   is_free    = state->budget > 0 && node.kind == OpKind::Free;
        size_t const free_bytes = is_free ? node.estimated_bytes : 0;

        pool.submit_detached(node.label, [state, &complete_node, &drain_deferred, &run_body, &node, i, is_free, free_bytes]() {
            run_body(node, [&node]() { execute_node(node); });
            if (is_free && free_bytes > 0) {
                state->mem_current.fetch_sub(free_bytes, std::memory_order_relaxed);
                drain_deferred();
            }
            complete_node(i);
        });
    };

    // Seed the dependency-free roots; everything else self-schedules.
    for (size_t i = 0; i < n; i++) {
        if (deps.predecessors[i].empty()) {
            submit_node(i);
        }
    }

    // The calling thread work-steals while waiting instead of parking.
    pool.help_until([&]() { return state->completed.load(std::memory_order_acquire) == n; });

    if (state->first_exc) {
        std::rethrow_exception(state->first_exc);
    }
}

// ─── MPIExecutor ────────────────────────────────────────────────────────────

void MPIExecutor::execute(Graph &graph) {
    // All ranks execute the same node sequence. Compute nodes operate on
    // local partitions; communication nodes are collective (all ranks
    // participate). On mock backend (1 rank), this is identical to Sequential.
    for (auto &node : graph.nodes()) {
        execute_timed(node, graph);
    }
}

} // namespace einsums::compute_graph
