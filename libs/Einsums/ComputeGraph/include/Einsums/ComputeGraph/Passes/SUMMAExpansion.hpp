//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief Distributed-execution pass: rewrites SUMMA-distributed einsums into broadcast+GEMM loops.
 *
 * @note Distributed (MPI) execution is a work in progress. This pass runs only on
 *       MPI-enabled (or mock) builds, and the distributed backend is not yet complete.
 *
 * When both inputs of an einsum are fully 2D-distributed with their contraction (link) index on the
 * grid, each rank's local block lacks the data for a direct GEMM. For nodes whose output
 * `DistributionDescriptor` has `summa == true` (and whose inputs are likewise SUMMA-distributed), this
 * pass replaces the einsum's executor with the SUMMA algorithm:
 *
 * 1. Apply the C prefactor (zero or scale `C_local`).
 * 2. For each of the `Pc` panels of the link dimension:
 *    - broadcast the A panel along `row_comm` (within the grid row),
 *    - broadcast the B panel along `col_comm` (within the grid column),
 *    - accumulate the local GEMM `C_local += A_panel * B_panel` via einsum dispatch.
 *
 * In `create_default()` this runs inside the distributed block
 * (`if constexpr (comm::has_mpi || comm::is_mock)`), after DistributionPlanning and Materialization and
 * right after @ref InputSlicing. It self-guards on `world_size() <= 1` and on a non-2D grid.
 *
 * @par Example (C++)
 * @code
 * // Run under MPI on a square grid, e.g. `mpirun -np 4 ./my_app` -> 2x2.
 * cg::Graph graph("summa");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::declare_tensor<double>("C", {N, N});
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);     // A, B SUMMA-distributed by DistributionPlanning
 * }
 * graph.apply(cg::PassManager::create_default());          // the einsum's executor becomes a SUMMA loop
 * @endcode
 *
 * @par Example (Python)
 * No standalone Python entry point: not individually exposed; runs inside the distributed pipeline via
 * `g.apply(cg.default_pass_manager())` and only fires on a square 2D MPI grid.
 *
 * @par Limitations
 * - A no-op on a single rank or a non-2D grid (`rows <= 1` or `cols <= 1`); exercised only under real
 *   MPI (or the mock backend with a genuine 2D grid).
 * - **Square grids only** (`Pr == Pc`): non-square grids are logged and skipped (K would split unevenly
 *   between A's and B's link dims), falling back to the outer-product strategy.
 * - **Rank-2 GEMM only**: nodes where C/A/B are not all rank 2 are skipped.
 * - **Real dtypes only**: only `float64` and `float32` are expanded; complex einsums are skipped.
 * - Only `Einsum` nodes are considered; `BatchedGemm` is intentionally ignored (distributed batched
 *   contractions are unsupported, see `docs/gemm_batching.rst`).
 * - Allocates fresh A/B panel buffers inside the executor on every call.
 *
 * @par Future improvements
 * - Rectangular-grid SUMMA (`Pr != Pc`) with correctly split panel dimensions.
 * - Complex-dtype and higher-rank contraction support.
 * - Reuse panel buffers across invocations instead of reallocating per call.
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
