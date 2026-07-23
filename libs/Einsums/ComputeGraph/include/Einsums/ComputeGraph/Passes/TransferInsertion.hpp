//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief GPU-offload transfer pass: inserts HostToDevice / DeviceToHost nodes around GPU-placed ops.
 *
 * @note GPU offload is a work in progress. This pass runs only on GPU-enabled (or mock)
 *       builds, and the transfer/execution backend it targets is not yet complete.
 *
 * Consumes the `Target::GPU` placement that @ref GPUPlacement produced and makes the graph
 * self-contained by materializing the copies each device op needs. For every GPU-targeted node:
 * - **Inputs**: if a tensor is not already Device- or Both-resident, prepend a HostToDevice node.
 *   H2D is skipped for "dead" inputs the op overwrites wholesale (einsum with `c_prefactor == 0`,
 *   permute with `beta == 0`), whose prior contents never reach the device.
 * - **Outputs**: if a later CPU node reads the result, append a DeviceToHost node.
 *
 * After the main loop it appends a final D2H for any user-visible (non-intermediate) tensor left
 * Device-resident, so results are readable after `execute()` on discrete GPUs (CUDA/HIP) without an
 * implicit flush. On unified memory (MPS) these D2H nodes stay in the structure but execution skips
 * the physical copy. `TensorHandle::residency` is updated as transfers are inserted (H2D → Both,
 * D2H → Both, GPU output → Device).
 *
 * In `create_default()` this runs inside the GPU-enabled block
 * (`if constexpr (gpu::has_gpu || gpu::is_mock)`), immediately after GPUPlacement and before
 * TransferElimination; the block is compiled out on CPU-only builds.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("transfer_insertion");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);     // placed on GPU by GPUPlacement
 * }
 * graph.apply(cg::PassManager::create_default());          // GPUPlacement -> TransferInsertion run in sequence
 * // With a GPU backend, H2D(A), H2D(B) now precede the einsum and a D2H(C) follows it.
 * @endcode
 *
 * @par Example (Python)
 * No standalone Python entry point: not individually exposed; runs inside the GPU pipeline.
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("transfer_insertion")
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", C, A, B)
 * g.apply(cg.default_pass_manager())                       # transfers are inserted only on a GPU-enabled build
 * @endcode
 *
 * @par Limitations
 * - Only meaningful when a node was actually placed on GPU; on CPU-only builds the GPU block is
 *   compiled out and nothing is inserted.
 * - The transfer executors are **placeholders**: `HostToDevice` / `DeviceToHost` currently only call
 *   `gpu::device_synchronize()`. Real byte copies await device-shadow allocations; on the mock and
 *   unified-memory (MPS) backends no separate device buffer exists, so the node is a sync point only.
 * - Loop-invariant inputs are re-transferred every iteration (the pass recurses into each body and
 *   makes it self-contained); hoisting invariant H2D out of the loop is not done here.
 * - Redundant transfers (e.g. an input already resident from a prior GPU op) are left for
 *   @ref TransferElimination to remove.
 *
 * @par Future improvements
 * - Replace the placeholder executors with real `gpu::memcpy_host_to_device` / D2H copies once
 *   device-shadow allocations land.
 * - Hoist loop-invariant H2D transfers out of loop bodies instead of re-issuing them per iteration.
 */
class EINSUMS_EXPORT TransferInsertion : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "TransferInsertion"; }
    bool                      run(Graph &graph) override;

    /// Recurse into loop bodies / conditional branches. TransferInsertion
    /// is a per-graph transform: run on a body it inserts that body's H2D
    /// before each GPU op and D2H after, making the body self-contained.
    /// It re-transfers loop-invariant inputs each iteration; hoisting those
    /// before the loop is a later optimization. See
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of transfer nodes inserted in the last run.
    [[nodiscard]] size_t num_transfers() const { return _num_transfers; }

  private:
    size_t _num_transfers{0};
};

} // namespace einsums::compute_graph::passes
