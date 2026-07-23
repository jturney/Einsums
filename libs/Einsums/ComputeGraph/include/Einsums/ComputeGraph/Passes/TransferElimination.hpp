//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief GPU-offload cleanup pass: removes redundant H2D/D2H transfers and inserts budget evictions.
 *
 * @rst
 * .. note::
 *
 *    GPU offload is a work in progress. This pass runs only on GPU-enabled (or mock)
 *    builds, and the transfer/execution backend it targets is not yet complete.
 * @endrst
 *
 * Simulates execution in topological order, tracking per-tensor residency and device-memory use, and
 * deletes transfer nodes that are already satisfied:
 * - a `HostToDevice` whose tensor is already Device- or Both-resident, and
 * - a `DeviceToHost` whose tensor is already Host- or Both-resident.
 * Consecutive GPU nodes sharing a tensor therefore avoid an intermediate D2H+H2D round-trip — the
 * tensor stays on device.
 *
 * When a required H2D would exceed `gpu::available_device_memory()`, it applies **Belady's optimal
 * strategy**: because the full schedule is known, the resident tensor whose next GPU use is furthest
 * away (and is not pinned by a loop body) is evicted with an inserted `D2H_evict` node until the
 * incoming tensor fits. Tensors used inside a loop body are pinned and never evicted.
 *
 * In `create_default()` this runs inside the GPU-enabled block
 * (`if constexpr (gpu::has_gpu || gpu::is_mock)`), immediately after @ref TransferInsertion; the block
 * is compiled out on CPU-only builds.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("transfer_elim");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &T, 1.0, A, B);     // T on GPU
 *     cg::einsum("il <- ij ; jl", 0.0, &C, 1.0, T, D);     // reuses T on GPU
 * }
 * graph.apply(cg::PassManager::create_default());          // TransferInsertion then TransferElimination
 * // The D2H(T)+H2D(T) round-trip between the two GPU einsums is removed; T stays resident.
 * @endcode
 *
 * @par Example (Python)
 * No standalone Python entry point: not individually exposed; runs inside the GPU pipeline.
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("transfer_elim")
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", T, A, B)
 *     einsums.einsum("il <- ij ; jl", C, T, D)
 * g.apply(cg.default_pass_manager())                       # redundant transfers pruned only on a GPU-enabled build
 * @endcode
 *
 * @par Limitations
 * - Only relevant after transfers exist; on CPU-only builds the GPU block is compiled out.
 * - Eviction uses `gpu::available_device_memory()` as the budget; on mock/unified-memory backends this
 *   is nominal, so eviction rarely triggers and cannot be validated without real device memory pressure.
 * - Pinning is coarse: any tensor touched by a loop body is pinned wholesale; if every resident tensor
 *   is pinned the pass cannot free space and simply proceeds (potential over-subscription on real hardware).
 * - Redundancy detection is residency-based simulation, not a full alias analysis.
 *
 * @par Future improvements
 * - Finer-grained eviction that pins only the tensors a loop actually keeps live across iterations.
 * - Exercise the Belady eviction path under real device budgets once device-shadow allocations land
 *   (today it is primarily reachable on hardware with a genuine memory ceiling).
 */
class EINSUMS_EXPORT TransferElimination : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "TransferElimination"; }
    bool                      run(Graph &graph) override;

    /// Recurse into loop bodies / conditional branches. Per-graph cleanup
    /// of redundant transfers within each body (its existing is_loop_tensor
    /// pin keeps tensors needed by a nested loop from being evicted).
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of transfer nodes removed in the last run.
    [[nodiscard]] size_t num_eliminated() const { return _num_eliminated; }

  private:
    size_t _num_eliminated{0};
};

} // namespace einsums::compute_graph::passes
