//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Plan allocation for deferred (shell) tensors by inserting nodes.
 *
 * For each tensor with AllocState::Deferred, this pass:
 * 1. Finds the earliest node that writes to or reads from this tensor.
 * 2. Inserts a Materialize node just before that point — its executor
 *    calls materialize_fn() to allocate the backing storage.
 * 3. If init_kind is set (Zero, Random), inserts an Initialize node
 *    immediately after the Materialize node.
 *
 * Allocation happens lazily during graph.execute() when the Materialize
 * node runs — NOT during the pass itself. This means:
 * - Intermediates only consume memory when needed
 * - MemoryPlanning can see Materialize nodes for accurate peak estimation
 * - Distributed allocation can happen at execution time per-rank
 *
 * Runs AFTER DistributionPlanningPass and BEFORE GPUPlacement.
 */
class EINSUMS_EXPORT Materialization : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "Materialization"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_materialized() const { return _num_materialized; }
    [[nodiscard]] size_t num_initialized() const { return _num_initialized; }

  private:
    size_t _num_materialized{0};
    size_t _num_initialized{0};
};

} // namespace einsums::compute_graph::passes
