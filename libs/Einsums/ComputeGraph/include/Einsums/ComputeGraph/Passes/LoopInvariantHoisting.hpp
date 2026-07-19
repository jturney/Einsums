//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Loop invariant hoisting pass.
 *
 * For each Loop node in the graph, identifies operations inside the loop body
 * whose inputs are never modified by any other operation in the loop body.
 * These operations are "loop-invariant", they compute the same result every
 * iteration and can be moved before the loop to execute only once.
 *
 * @par Example
 * @code
 * // Before hoisting:
 * loop {
 *     T = X^T * H    // X and H never change in the loop → invariant!
 *     F = T * X       // T is computed above, but T depends only on invariants
 *     D = f(F)        // D depends on F which changes → NOT invariant
 * }
 *
 * // After hoisting:
 * T = X^T * H         // Hoisted out of loop
 * F = T * X           // Hoisted out of loop
 * loop {
 *     D = f(F)         // Stays in loop
 * }
 * @endcode
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT LoopInvariantHoisting : public OptimizerPass {
  public:
    APIARY_EXPOSE LoopInvariantHoisting() = default;

    [[nodiscard]] std::string name() const override { return "LoopInvariantHoisting"; }
    bool                      run(Graph &graph) override;

    APIARY_EXPOSE APIARY_GETTER("num_hoisted") [[nodiscard]] size_t num_hoisted() const { return _num_hoisted; }

  private:
    /// Innermost-first descent. Recurse into every Loop body that is a direct
    /// child of @p graph BEFORE hoisting at this level (Conditional branches are
    /// deliberately NOT descended, see the .cpp). A producer invariant across
    /// several nested loops is lifted one level per composed single-level hoist.
    void run_recursive(Graph &graph);

    /// Single-level driver: hoist invariant producers out of each Loop body that
    /// is a direct child of @p graph, up one level into @p graph. Accumulates
    /// into @ref _num_hoisted and re-sorts @p graph if it hoisted anything.
    void hoist_one_level(Graph &graph);

    size_t _num_hoisted{0};
};

} // namespace einsums::compute_graph::passes
