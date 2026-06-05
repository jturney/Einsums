//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file Executor.hpp
 * @brief Pluggable execution backends for computation graphs.
 *
 * The Executor abstraction decouples the graph structure from the execution
 * strategy. Built-in executors:
 * - **SequentialExecutor**: Executes nodes in topological order (default)
 * - **OpenMPExecutor**: Launches independent nodes as OpenMP tasks
 *
 * Future executors: CudaStreamExecutor, CoroutineExecutor.
 *
 * @par Example
 * @code
 * cg::OpenMPExecutor omp_exec;
 * graph.execute(omp_exec);           // Parallel execution
 * graph.execute();                   // Sequential (default)
 * pipeline.execute(omp_exec);       // Pipeline with parallel stages
 * @endcode
 */

#include <Einsums/Config.hpp>

#include <Einsums/Python/Annotations.hpp>

#include <string>
#include <vector>

namespace einsums::compute_graph {

class Graph; // Forward declaration

/**
 * @brief Dependency information for a sorted graph.
 *
 * Computed by Graph::topological_sort() and stored for reuse by executors.
 * Indices refer to positions in the sorted nodes vector.
 */
struct DependencyInfo {
    std::vector<std::vector<size_t>> successors;   ///< successors[i] = nodes that depend on node i
    std::vector<std::vector<size_t>> predecessors; ///< predecessors[i] = nodes that node i depends on
};

/**
 * @brief Abstract base class for graph execution backends.
 *
 * An executor determines HOW graph nodes are executed — sequentially,
 * in parallel via threads, on GPU streams, etc. The graph structure,
 * optimization passes, and capture mechanism are executor-agnostic.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_NOCOPY APIARY_NOMOVE Executor {
  public:
    virtual ~Executor() = default;

    APIARY_EXPOSE [[nodiscard]] virtual std::string name() const = 0;

    /**
     * @brief Execute all nodes of the graph.
     *
     * The graph is guaranteed to be topologically sorted with valid
     * dependency info before this is called.
     *
     * @param[in,out] graph The graph to execute.
     */
    virtual void execute(Graph &graph) = 0;
};

/**
 * @brief Sequential executor — executes nodes in topological order.
 *
 * This is the default execution strategy. Nodes run one at a time
 * in dependency order. Zero overhead, no thread safety concerns.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_NOCOPY APIARY_NOMOVE EINSUMS_EXPORT SequentialExecutor : public Executor {
  public:
    APIARY_EXPOSE SequentialExecutor() = default;

    [[nodiscard]] std::string name() const override { return "Sequential"; }
    void                      execute(Graph &graph) override;
};

/**
 * @brief OpenMP task-based executor — launches independent nodes in parallel.
 *
 * Uses OpenMP tasks with dependency tracking. When a node completes,
 * it decrements the in-degree of its successors. Successors whose
 * in-degree reaches zero are launched as new tasks.
 *
 * @note BLAS calls within nodes may use their own OpenMP parallelism.
 *       For small graphs, the sequential executor may be faster due
 *       to reduced thread management overhead.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_NOCOPY APIARY_NOMOVE EINSUMS_EXPORT OpenMPExecutor : public Executor {
  public:
    APIARY_EXPOSE OpenMPExecutor() = default;

    [[nodiscard]] std::string name() const override { return "OpenMP"; }
    void                      execute(Graph &graph) override;
};

/**
 * @brief TaskPool dataflow executor — maximum overlap via continuations.
 *
 * Uses the TaskPool module to execute graph nodes as dataflow tasks.
 * Each node is submitted when all its predecessors complete, using
 * TaskHandle continuations for zero-barrier scheduling.
 *
 * Independent nodes run concurrently across TaskPool workers. The
 * work-stealing deque provides automatic load balancing.
 *
 * @note Requires the TaskPool module. Falls back to SequentialExecutor
 *       if TaskPool is not available.
 *
 * @par Example
 * @code
 * cg::DataflowExecutor df_exec;
 * graph.execute(df_exec);  // Maximum overlap via TaskPool
 * @endcode
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_NOCOPY APIARY_NOMOVE EINSUMS_EXPORT DataflowExecutor : public Executor {
  public:
    APIARY_EXPOSE DataflowExecutor() = default;

    [[nodiscard]] std::string name() const override { return "Dataflow"; }
    void                      execute(Graph &graph) override;

    /**
     * @brief Set a memory budget for execution.
     *
     * When set (> 0), the executor tracks live tensor memory and gates
     * Materialize node submissions to stay within the budget. Free nodes
     * release memory and wake blocked submissions.
     *
     * @param bytes Maximum bytes of simultaneously live tensor data. 0 = unlimited (default).
     */
    APIARY_EXPOSE void set_memory_budget(size_t bytes) { _memory_budget = bytes; }

    /// Get the current memory budget (0 = unlimited).
    APIARY_EXPOSE [[nodiscard]] size_t memory_budget() const { return _memory_budget; }

  private:
    size_t _memory_budget{0};
};

/**
 * @brief MPI-aware executor for distributed computation graphs.
 *
 * All ranks execute the same graph. Compute nodes operate on local
 * partitions; communication nodes (Allreduce, Broadcast, Barrier)
 * are collective operations that all ranks participate in.
 *
 * On the mock backend (single rank), behaves identically to
 * SequentialExecutor — communication nodes are no-ops.
 *
 * @par Example
 * @code
 * cg::MPIExecutor mpi_exec;
 * graph.execute(mpi_exec);  // All ranks execute the graph
 * @endcode
 */
class MPIExecutor : public Executor {
  public:
    [[nodiscard]] std::string name() const override { return "MPI"; }
    void                      execute(Graph &graph) override;
};

} // namespace einsums::compute_graph
