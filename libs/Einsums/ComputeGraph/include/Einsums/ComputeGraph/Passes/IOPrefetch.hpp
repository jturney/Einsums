//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief IOPrefetch pass: move DiskRead nodes as early as legally possible in the schedule.
 *
 * For each DiskRead node, computes its earliest legal position - after every producer of its inputs and
 * after any prior node that reads or writes its destination tensor (moving a load before a node that
 * touches its destination would be a WAR/WAW violation) - and moves it there. This maximizes the window
 * between a read's async_start and its first consumer, enabling maximum I/O-compute overlap when used
 * with the DataflowExecutor. A loop-invariant DiskRead inside a loop body (destination written exactly
 * once across the whole loop subtree, untouched by nested sub-graphs, already materialized) is HOISTED
 * out before the loop, one nesting level per recursion step, so it reads once instead of per iteration.
 *
 * DiskWrite nodes are NOT moved - writes should happen as late as possible to avoid blocking compute on
 * I/O completion.
 *
 * This pass is in the default pipeline, after Reorder (which optimizes node order for memory, so
 * IOPrefetch can pull reads earlier without violating dependency constraints) and before
 * DistributionPlanning / Materialization.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("prefetch");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::scale(2.0, &A);                                   // compute, unrelated to the load
 *     cg::scale(3.0, &A);
 *     cg::read("load data", "data.h5", "/data", &data, load_fn);   // no dependency on the scales
 *     cg::axpy(1.0, data, &C);                              // first consumer of the load
 * }
 * graph.apply(cg::PassManager::create_default());           // IOPrefetch runs here
 * // The DiskRead now sits at the front, overlapping its I/O with the scales under the DataflowExecutor.
 * @endcode
 *
 * @par Example (Python)
 * Pipeline-internal: not exposed for standalone construction in Python. Disk I/O reads captured into the
 * graph are prefetched by the default manager.
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("prefetch")
 * with cg.capture(g):
 *     einsums.linalg.scale(2.0, A)
 *     # ... a captured DiskRead of `data`, then a consumer of it ...
 * g.apply(cg.default_pass_manager())                        # moves the DiskRead as early as legal
 * @endcode
 *
 * @par Limitations
 * - Only DiskRead nodes are moved; DiskWrite nodes are deliberately left in place (writes stay late).
 * - Within-graph movement respects both the read's input producers and any prior node that touches its
 *   destination tensor; a read can never pass a node that reads or writes what it loads into.
 * - Loop-invariant hoisting fires only when the destination has exactly one writer (the read itself)
 *   across the entire loop subtree AND is already materialized (eager). Because IOPrefetch runs before
 *   Materialization, a deferred-shell destination keeps its per-iteration read rather than being hoisted.
 *
 * @par Future improvements
 * - A deferred-destination read cannot be hoisted here (its storage is not yet allocated); a second
 *   prefetch/hoist opportunity after Materialization could cover those loop-invariant loads.
 */
class EINSUMS_EXPORT IOPrefetch : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "IOPrefetch"; }
    bool                      run(Graph &graph) override;

    /// Number of DiskRead nodes moved in the last run.
    [[nodiscard]] size_t num_prefetched() const { return _num_prefetched; }

  private:
    size_t _num_prefetched{0};
};

} // namespace einsums::compute_graph::passes
