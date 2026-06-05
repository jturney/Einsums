//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Constant folding pass.
 *
 * Identifies nodes whose inputs are all "constant" — meaning no other node
 * in the graph writes to those tensors. These nodes can be executed once
 * during the pass and replaced with no-ops, since their results won't
 * change across graph replays.
 *
 * Only graph-owned intermediate tensors (created via `create_tensor()`) that
 * are never written by any node are treated as initially constant. User-owned
 * tensors are NOT assumed constant because they may change between loop
 * iterations or between successive `execute()` calls. This makes the pass
 * safe for both one-shot graphs and loop bodies.
 *
 * This is especially useful for setup operations that build transformation
 * matrices, orthogonalization matrices, etc. from fixed input data.
 *
 * @par Example
 * @code
 * // If A and B are never written by any node in the graph:
 * cg::einsum("ij <- ik ; kj", &C, A, B);  // Constant! Fold it.
 * cg::scale(2.0, &C);                      // C is written above, so this depends on it
 *                                           // But after folding, C is constant too.
 * @endcode
 *
 * @note The pass executes folded nodes during the pass itself (not during
 *       graph.execute()). This means the pass has side effects on tensor data.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ConstantFolding : public OptimizerPass {
  public:
    APIARY_EXPOSE ConstantFolding() = default;

    [[nodiscard]] std::string name() const override { return "ConstantFolding"; }
    bool                      run(Graph &graph) override;

    /// Recurse into loop bodies / conditional branches. Safe now that
    /// run() only folds nodes whose tensors are materialized at pass time
    /// (see the materialization guard in run()): a loop body's deferred
    /// workspace tensors are simply skipped rather than executed against
    /// unallocated storage. Loop-carried tensors are never folded because
    /// they appear in ``written_tensors``. See docs/loop_handling_audit.md.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of nodes folded in the last run.
    APIARY_EXPOSE APIARY_GETTER("num_folded") [[nodiscard]] size_t num_folded() const { return _num_folded; }

  private:
    size_t _num_folded{0};
};

} // namespace einsums::compute_graph::passes
