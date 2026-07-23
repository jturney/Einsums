//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Distributed-execution pass: inserts collective communication where partial results need combining.
 *
 * @rst
 * .. note::
 *
 *    Distributed (MPI) execution is a work in progress. This pass runs only on
 *    MPI-enabled (or mock) builds, and the distributed backend is not yet complete.
 * @endrst
 *
 * The distributed analogue of @ref TransferInsertion. Walks the graph and, for any compute node
 * (Einsum, Scale, Axpy, …) that reads a distributed (non-replicated) input but writes a replicated
 * output, inserts an in-place `Allreduce` immediately after it — the output holds a per-rank partial
 * sum that must be combined into the full result. The allreduce executor is the tensor handle's
 * type-erased `allreduce_sum_fn`.
 *
 * @note **Ordering semantics**: the Allreduce is placed right after the producing node, so element-wise
 * ops captured later operate on the globally-reduced result — correct for the common pattern:
 * @code
 * einsum(C = A*B);   // partial sum on each rank
 * // [allreduce(C) inserted here]
 * scale(2.0, C);     // operates on the full, reduced result
 * @endcode
 *
 * In `create_default()` this runs inside the distributed block
 * (`if constexpr (comm::has_mpi || comm::is_mock)`), after @ref SUMMAExpansion and before
 * @ref CommunicationElimination. It self-guards with `world_size() <= 1` and is a no-op on a single rank.
 *
 * @par Example (C++)
 * @code
 * // Run under MPI, e.g. `mpirun -np 4 ./my_app`.
 * cg::Graph graph("comm_insertion");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);     // A distributed on the link index -> partial C
 * }
 * graph.apply(cg::PassManager::create_default());          // allreduce(C) is inserted after the einsum
 * @endcode
 *
 * @par Example (Python)
 * No standalone Python entry point: not individually exposed; runs inside the distributed pipeline via
 * `g.apply(cg.default_pass_manager())` and only fires on more than one MPI rank.
 *
 * @par Limitations
 * - A no-op on a single rank; exercised only under real MPI (or the mock backend with `world_size() > 1`).
 * - Only inserts **Allreduce** (partial-sum → full). Broadcast/Allgather/Scatter are recognized as
 *   existing infrastructure nodes but are not synthesized here.
 * - Infrastructure and control-flow nodes (Materialize, Initialize, transfers, existing collectives,
 *   Loop, Conditional, …) are skipped; distributed reductions inside a loop body are not inserted by
 *   this top-level walk.
 * - Requires the tensor handle to carry a valid `allreduce_sum_fn`; otherwise the inserted node is a no-op.
 * - `BatchedGemm` is not handled (batched + distributed is unsupported, `docs/gemm_batching.rst`).
 *
 * @par Future improvements
 * - Synthesize Broadcast/Allgather for the layouts that need them, not just Allreduce.
 * - Recurse into loop bodies / conditional branches so distributed reductions inside them are inserted.
 */
class EINSUMS_EXPORT CommunicationInsertion : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "CommunicationInsertion"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_inserted() const { return _num_inserted; }

  private:
    size_t _num_inserted{0};
};

} // namespace einsums::compute_graph::passes
