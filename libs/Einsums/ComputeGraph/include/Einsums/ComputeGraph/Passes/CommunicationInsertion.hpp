//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Insert communication nodes (Allreduce, Broadcast, Allgather) where needed.
 *
 * Analogous to TransferInsertion for GPU passes. Walks the graph and inserts
 * collective communication nodes when a distributed tensor is produced by one
 * node and consumed by another that needs the full (non-partial) result.
 *
 * Inserts allreduce nodes after any compute node (Einsum, Scale, Axpy, etc.)
 * that has distributed inputs and replicated outputs (partial sum → full result).
 *
 * @note **Ordering semantics**: Allreduce is inserted immediately after the
 * producing compute node. Element-wise operations (scale, axpy) captured
 * after the einsum will execute AFTER the allreduce, operating on the
 * globally-reduced result. This is correct for the common pattern:
 *   einsum(C = A*B);  // partial sum on each rank
 *   // [allreduce(C) inserted here]
 *   scale(2.0, C);    // operates on full result
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
