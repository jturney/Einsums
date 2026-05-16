//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Generalized scale absorption pass.
 *
 * Absorbs Scale(α, C) into any subsequent operation that writes to C with a
 * beta/c_prefactor parameter:
 *
 * - **Einsum**: Scale(α) + Einsum(c_pf=0) → Einsum(c_pf=α)
 * - **Gemm**: Scale(α) + Gemm(beta=0) → Gemm(beta=α)
 * - **Permute**: Scale(α) + Permute(beta=0) → Permute(beta=α)
 *
 * The Scale node is removed from the graph and its effect is absorbed
 * into the following operation's prefactor.
 */
class EINSUMS_EXPORT ScaleAbsorption : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "ScaleAbsorption"; }
    bool                      run(Graph &graph) override;

    /// Number of scale nodes absorbed in the last run.
    [[nodiscard]] size_t num_absorbed() const { return _num_absorbed; }

  private:
    size_t _num_absorbed{0};
};

} // namespace einsums::compute_graph::passes
