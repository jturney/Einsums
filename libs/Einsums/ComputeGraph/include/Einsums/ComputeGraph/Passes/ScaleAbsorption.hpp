//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Dead-scale elimination pass.
 *
 * Removes Scale(α, C) when the next node writing C overwrites it without
 * reading its prior contents (c_prefactor / beta == 0) and no node reads C
 * in between — the scale's result is discarded wholesale, so the scal is
 * pure wasted work:
 *
 * - **Einsum**: Scale(α) + Einsum(c_pf=0) → Einsum(c_pf=0)
 * - **BatchedGemm**: Scale(α) + BatchedGemm(beta=0) → BatchedGemm(beta=0)
 * - **Permute**: Scale(α) + Permute(beta=0) → Permute(beta=0)
 *
 * The following operation is left untouched. It must not be rewritten to
 * "absorb" the factor: CPU einsum executors read prefactors live from the
 * shared EinsumParams while GPU dispatch reads the descriptor, so a
 * descriptor-only edit desyncs the backends (and the scaled value is dead
 * anyway — absorbing it would change the result).
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ScaleAbsorption : public OptimizerPass {
  public:
    APIARY_EXPOSE ScaleAbsorption() = default;

    [[nodiscard]] std::string name() const override { return "ScaleAbsorption"; }
    bool                      run(Graph &graph) override;

    /// Safe on loop bodies / conditional branches: a local rewrite that
    /// only folds scale into the next op within the graph it's handed.
    /// See docs/loop_handling_audit.md.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of scale nodes absorbed in the last run.
    APIARY_EXPOSE APIARY_GETTER("num_absorbed") [[nodiscard]] size_t num_absorbed() const { return _num_absorbed; }

  private:
    size_t _num_absorbed{0};
};

} // namespace einsums::compute_graph::passes
