//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief Distribution-planning pass: chooses a process-grid layout for each deferred tensor.
 *
 * @note Distributed (MPI) execution is a work in progress. This pass runs only on
 *       MPI-enabled (or mock) builds, and the distributed backend is not yet complete.
 *
 * For every tensor with `AllocState::Deferred`, decides between replication and 2D block
 * distribution over the `comm::ProcessGrid`, then annotates its `TensorHandle`
 * (`is_distributed`, `is_replicated`, `distribution_info`). @ref Materialization runs afterward and
 * reads these to allocate either the full tensor or just the local partition.
 *
 * The layout is derived from how the tensor is used in an einsum: output indices coming from A map to
 * grid **Row**s, those from B to grid **Col**s, shared/batch indices balance across the axis with
 * fewer assignments, and (when SUMMA is enabled on a square grid) contraction "link" indices are
 * distributed too — A's link → Col, B's link → Row. Two consistency passes then run: **conflict
 * resolution** downgrades a dim to `None` if the same index is a contraction index in another einsum,
 * and **constraint propagation** makes a consumer inherit an already-distributed input's axis so chain
 * intermediates stay consistent. Tensors below `threshold` bytes, or used by no classifiable node, are
 * replicated. On a single rank (`world_size() <= 1`) the pass is a no-op — everything stays replicated
 * at full size.
 *
 * Unlike the other distributed passes, this one is added **unconditionally** in `create_default()`,
 * just before Materialization (it must set layouts before allocation), rather than inside the
 * `if constexpr (comm::has_mpi || comm::is_mock)` block. It still self-guards on rank count at runtime.
 *
 * @par Example (C++)
 * @code
 * // Run under MPI, e.g. `mpirun -np 4 ./my_app`.
 * cg::Graph graph("dist_plan");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::declare_tensor<double>("C", {N, N});             // deferred output
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);
 * }
 * graph.apply(cg::PassManager::create_default());          // DistributionPlanning precedes Materialization
 * // On >1 rank a large C is block-distributed on the grid; on 1 rank it is replicated (no-op).
 * @endcode
 *
 * @par Example (Python)
 * No standalone Python entry point: not individually exposed; runs inside the pipeline via
 * `g.apply(cg.default_pass_manager())` and only distributes when launched on more than one MPI rank.
 *
 * @par Limitations
 * - Distributes only **deferred** tensors; pre-allocated inputs are handled by @ref InputSlicing.
 * - A no-op on a single rank, so distribution logic is exercised only under real MPI (or the mock
 *   backend with `world_size() > 1`).
 * - Classifies only `Einsum` and `Permute`/`Transpose` nodes; `BatchedGemm` is skipped because mixing
 *   GEMM batching with the distribution pipeline is unsupported (see `docs/gemm_batching.rst`).
 * - SUMMA link distribution requires a true **square** 2D grid; on rectangular grids link dims are left
 *   `None` and the contraction falls back to the outer-product strategy.
 * - Conflict resolution and constraint propagation match dimensions **positionally**, assuming the two
 *   operands' index lists line up index-for-index at each dim.
 *
 * @par Future improvements
 * - Rectangular-grid SUMMA layouts (today only `Pr == Pc` distributes link indices).
 * - Broaden classification beyond einsum/permute so more op kinds can participate in distribution.
 */
class EINSUMS_EXPORT DistributionPlanning : public OptimizerPass {
  public:
    /// @param threshold Tensors smaller than this (bytes) are replicated.
    /// @param enable_summa If true (default), distribute link indices for SUMMA on square grids.
    explicit DistributionPlanning(size_t threshold = static_cast<size_t>(64 * 1024 * 1024), bool enable_summa = true);

    [[nodiscard]] std::string name() const override { return "DistributionPlanning"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_distributed() const { return _num_distributed; }
    [[nodiscard]] size_t num_replicated() const { return _num_replicated; }

  private:
    size_t _threshold;
    bool   _enable_summa;
    size_t _num_distributed{0};
    size_t _num_replicated{0};
};

} // namespace einsums::compute_graph::passes
