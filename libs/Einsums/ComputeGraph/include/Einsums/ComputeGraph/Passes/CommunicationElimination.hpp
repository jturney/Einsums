//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Distributed-execution cleanup pass: removes redundant communication nodes.
 *
 * @note Distributed (MPI) execution is a work in progress. This pass runs only on
 *       MPI-enabled (or mock) builds, and the distributed backend is not yet complete.
 *
 * The distributed analogue of @ref TransferElimination. Walks the node list in order, tracking which
 * tensors have already been reduced, and deletes a second `Allreduce` of a tensor that has not been
 * written since its previous reduce. Any non-collective write to a tensor invalidates its
 * "already reduced" status, so a genuine recompute-then-reduce is preserved.
 *
 * In `create_default()` this runs inside the distributed block
 * (`if constexpr (comm::has_mpi || comm::is_mock)`), immediately after @ref CommunicationInsertion and
 * before @ref CommunicationScheduling.
 *
 * @par Example (C++)
 * @code
 * // Run under MPI, e.g. `mpirun -np 4 ./my_app`.
 * cg::Graph graph("comm_elim");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);     // -> allreduce(C) inserted
 *     cg::scale(2.0, &C);                                  // reads reduced C; no rewrite of C's reduction
 * }
 * graph.apply(cg::PassManager::create_default());          // a duplicate allreduce(C) with no intervening write is removed
 * @endcode
 *
 * @par Example (Python)
 * No standalone Python entry point: not individually exposed; runs inside the distributed pipeline via
 * `g.apply(cg.default_pass_manager())`.
 *
 * @par Limitations
 * - Only eliminates redundant **Allreduce** nodes today; the "broadcast of an already-replicated tensor" and
 *   "allgather when everyone has the data" cases described for the family are not yet detected.
 * - Redundancy is decided by a single top-level forward walk with a write-invalidation set — no
 *   cross-loop-body or alias-aware reasoning.
 * - Unlike the counting-only distributed passes, this one does not self-guard on rank count (it simply
 *   finds no redundant collectives when the insertion pass produced none on a single rank).
 *
 * @par Future improvements
 * - Detect and drop redundant Broadcast and Allgather nodes, matching the family's intent.
 * - Alias- and loop-aware redundancy analysis rather than a flat forward scan.
 */
class EINSUMS_EXPORT CommunicationElimination : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "CommunicationElimination"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_eliminated() const { return _num_eliminated; }

  private:
    size_t _num_eliminated{0};
};

} // namespace einsums::compute_graph::passes
