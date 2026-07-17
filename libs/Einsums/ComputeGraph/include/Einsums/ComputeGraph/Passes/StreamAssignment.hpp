//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief Assign nodes to execution streams for potential async overlap.
 *
 * Transfer nodes (H2D, D2H) are assigned to the transfer stream (stream_id=1).
 * Compute nodes (GPU-targeted Einsum, BLAS, etc.) stay on the compute stream (stream_id=0).
 * CPU nodes stay on stream 0.
 *
 * This enables future double-buffering: H2D for tensor N+1 can overlap with
 * compute of tensor N when they're on different streams.
 *
 * Currently execution is synchronous regardless of stream assignment. The
 * assignments are structural metadata for when async execution is implemented.
 */
class EINSUMS_EXPORT StreamAssignment : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "StreamAssignment"; }
    bool                      run(Graph &graph) override;

    /// Safe on loop bodies / conditional branches: assigns a stream id to
    /// each node based purely on its kind, scoped to the graph it's
    /// handed.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    [[nodiscard]] size_t num_assigned() const { return _num_assigned; }

  private:
    size_t _num_assigned{0};
};

} // namespace einsums::compute_graph::passes
