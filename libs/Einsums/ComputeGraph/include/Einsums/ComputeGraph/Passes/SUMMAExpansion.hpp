//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief Replace einsum nodes with SUMMA-style broadcast+GEMM loops.
 *
 * When both inputs of an einsum are fully 2D-distributed on the process grid
 * (with link indices on the grid), the local partition doesn't have enough data
 * for a direct GEMM. This pass rewrites the einsum's executor to perform the
 * SUMMA algorithm:
 *
 * 1. Loop over Pc panels of the link dimension
 * 2. Broadcast A panel along row_comm (within grid row)
 * 3. Broadcast B panel along col_comm (within grid column)
 * 4. Local GEMM: C_local += A_panel * B_panel
 *
 * Runs AFTER DistributionPlanning and Materialization.
 * Only modifies nodes where distribution_info has summa=true.
 */
class EINSUMS_EXPORT SUMMAExpansion : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "SUMMAExpansion"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_expanded() const { return _num_expanded; }

  private:
    size_t _num_expanded{0};
};

} // namespace einsums::compute_graph::passes
