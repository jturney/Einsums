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

    // An exception must NOT escape an OpenMP structured block — doing so leaves
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

    auto      &pool = task_pool::TaskPool::get_singleton();
    std::mutex timing_mutex;

    // Memory budget tracking (shared across all tasks)
    struct MemBudgetState {
        std::atomic<size_t>     current{0};
        std::mutex              gate_mutex;
        std::condition_variable gate_cv;
    };
    auto   mem_state = std::make_shared<MemBudgetState>();
    size_t budget    = _memory_budget;

    std::vector<task_pool::TaskHandle<void>> handles(n);

    for (size_t i = 0; i < n; i++) {
        auto const &preds = deps.predecessors[i];

        // Build predecessor dependency handles.
        std::vector<task_pool::TaskHandle<void>> pred_handles;
        pred_handles.reserve(preds.size());
        for (size_t const p : preds) {
            pred_handles.push_back(handles[p]);
        }

        bool const has_async = static_cast<bool>(nodes[i].async_start) && static_cast<bool>(nodes[i].async_finish);

        // Memory budget gating: if this is a Materialize node and budget is set,
        // wait until there's enough room before submitting.
        if (budget > 0 && nodes[i].kind == OpKind::Materialize) {
            size_t alloc_bytes = nodes[i].estimated_bytes;
            if (alloc_bytes > 0) {
                std::unique_lock lock(mem_state->gate_mutex);
                mem_state->gate_cv.wait(lock, [&]() { return mem_state->current.load() + alloc_bytes <= budget; });
                mem_state->current += alloc_bytes;
            }
        }

        if (has_async) {
            task_pool::TaskHandle<void> start_handle;
            if (preds.empty()) {
                start_handle = pool.submit(nodes[i].label + "/start", [&node = nodes[i]]() { node.async_start(); });
            } else {
                auto all_preds = task_pool::when_all(std::move(pred_handles));
                start_handle   = pool.dataflow(
                    nodes[i].label + "/start", [&node = nodes[i]]() { node.async_start(); }, all_preds);
            }

            handles[i] = pool.dataflow(
                nodes[i].label + "/finish",
                [&node = nodes[i], &graph, &timing_mutex]() {
                    auto t0 = std::chrono::steady_clock::now();
                    node.async_finish();
                    auto                   t1 = std::chrono::steady_clock::now();
                    double const           ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    std::scoped_lock const lock(timing_mutex);
                    graph.record_node_timing(node.id, node.label, node.kind, ms);
                },
                start_handle);
        } else {
            // For Free nodes with budget tracking: decrement memory and wake the gate
            bool const   is_free    = (budget > 0 && nodes[i].kind == OpKind::Free);
            size_t const free_bytes = is_free ? nodes[i].estimated_bytes : 0;

            auto task_fn = [&node = nodes[i], &graph, &timing_mutex, mem_state, is_free, free_bytes]() {
                execute_timed_mt(node, graph, timing_mutex);
                if (is_free && free_bytes > 0) {
                    mem_state->current -= free_bytes;
                    mem_state->gate_cv.notify_one();
                }
            };

            if (preds.empty()) {
                handles[i] = pool.submit(nodes[i].label, std::move(task_fn));
            } else {
                auto all_preds = task_pool::when_all(std::move(pred_handles));
                handles[i]     = pool.dataflow(nodes[i].label, std::move(task_fn), all_preds);
            }
        }
    }

    for (size_t i = 0; i < n; i++) {
        if (deps.successors[i].empty()) {
            handles[i].wait();
        }
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
