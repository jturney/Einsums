//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief StreamAssignment pass: assign nodes to execution streams for potential async overlap.
 *
 * @rst
 * .. note::
 *
 *    GPU offload is a work in progress. This pass runs only on GPU-enabled (or mock)
 *    builds, and execution is still synchronous (the stream tags are not yet acted on).
 * @endrst
 *
 * Transfer nodes (HostToDevice, DeviceToHost) are assigned to the transfer stream (stream_id = 1); all
 * other nodes - GPU-targeted Einsum/BLAS compute and CPU nodes alike - stay on the compute stream
 * (stream_id = 0). The assignment is purely by node kind, per graph, and recurses into loop bodies /
 * conditional branches. This enables future double-buffering: an H2D for tensor N+1 can overlap with the
 * compute of tensor N once they sit on different streams.
 *
 * Currently execution is synchronous regardless of stream assignment. The assignments are structural
 * metadata for when async execution is implemented.
 *
 * This pass is in the default pipeline only when a GPU backend (or the mock GPU) is available, where it
 * runs as the last of the GPU passes (after GPUPlacement / TransferInsertion / TransferElimination /
 * GPUDiagnostics). On CPU-only builds it is not added.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("stream_assignment");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);      // GPUPlacement may target this to the GPU
 * }
 * graph.apply(cg::PassManager::create_default());           // on a GPU/mock build, StreamAssignment runs
 * // Any inserted H2D/D2H transfer nodes now carry stream_id = 1; compute stays on stream 0.
 * @endcode
 *
 * @par Example (Python)
 * Pipeline-internal (GPU builds only): not exposed for standalone construction in Python. Stream ids are
 * assigned by the default manager when a GPU/mock backend is present.
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("stream_assignment")
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", C, A, B)
 * g.apply(cg.default_pass_manager())                        # assigns transfer nodes to the transfer stream
 * @endcode
 *
 * @par Limitations
 * - Purely structural metadata: it only writes stream_id (H2D/D2H -> 1, everything else -> 0). Execution
 *   is synchronous today, so the assignment produces no actual overlap yet.
 * - Assignment is by node kind alone - there is no dependency-aware scheduling, no per-device streams,
 *   and no finer partitioning of independent compute across multiple streams.
 * - Added to the default pipeline only on GPU (or mock-GPU) builds; a CPU-only build never runs it.
 *
 * @par Future improvements
 * - Implement genuine asynchronous, double-buffered execution so an H2D for tensor N+1 overlaps the
 *   compute of tensor N - the reason the stream ids are assigned in the first place.
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
