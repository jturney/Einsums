//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>
#include <iosfwd>

namespace einsums::compute_graph::passes {

/**
 * @brief GPU-offload diagnostic pass: summarizes placement and transfers without mutating the graph.
 *
 * @note GPU offload is a work in progress. This pass runs only on GPU-enabled (or mock)
 *       builds, and the transfer/execution backend it targets is not yet complete.
 *
 * A read-only reporting pass. It walks the whole graph tree (loop bodies and conditional branches
 * included) and aggregates:
 * - nodes placed on GPU vs CPU,
 * - HostToDevice and DeviceToHost transfer counts,
 * - total bytes transferred, and
 * - peak estimated device-memory usage (maxed across graphs; residency is not carried across a
 *   loop boundary in this model).
 *
 * `run()` logs the summary and always returns `false` (no modification); `print_report()` renders the
 * same figures to a stream. Getters expose each counter for tests and tooling.
 *
 * In `create_default()` this runs inside the GPU-enabled block
 * (`if constexpr (gpu::has_gpu || gpu::is_mock)`), after GPUPlacement / TransferInsertion /
 * TransferElimination so it observes the final placement; the block is compiled out on CPU-only builds.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("gpu_diag");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);
 * }
 * graph.apply(cg::PassManager::create_default());          // GPUDiagnostics is the last GPU pass
 * // Counters are logged; call print_report(std::cout) for a formatted breakdown.
 * @endcode
 *
 * @par Example (Python)
 * No standalone Python entry point: not individually exposed; the summary is logged when the GPU
 * pipeline runs via `g.apply(cg.default_pass_manager())` on a GPU-enabled build.
 *
 * @par Limitations
 * - Purely informational — changes nothing and cannot influence placement.
 * - Peak device memory is an estimate from tensor byte sizes and per-graph residency, not a measured
 *   allocator high-water mark; it is maxed (not summed) across loop bodies.
 * - Produces meaningful numbers only after the mutating GPU passes have run; on CPU-only builds the
 *   GPU block is compiled out and the pass is never scheduled.
 *
 * @par Future improvements
 * - Report measured (allocator-level) device usage once real device-shadow allocations exist.
 */
class EINSUMS_EXPORT GPUDiagnostics : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "GPUDiagnostics"; }
    bool                      run(Graph &graph) override;

    /// Print the diagnostic report to a stream.
    void print_report(std::ostream &os) const;

    [[nodiscard]] size_t gpu_nodes() const { return _gpu_nodes; }
    [[nodiscard]] size_t cpu_nodes() const { return _cpu_nodes; }
    [[nodiscard]] size_t h2d_transfers() const { return _h2d_transfers; }
    [[nodiscard]] size_t d2h_transfers() const { return _d2h_transfers; }
    [[nodiscard]] size_t total_transfer_bytes() const { return _total_transfer_bytes; }
    [[nodiscard]] size_t peak_device_bytes() const { return _peak_device_bytes; }

  private:
    size_t _gpu_nodes{0};
    size_t _cpu_nodes{0};
    size_t _h2d_transfers{0};
    size_t _d2h_transfers{0};
    size_t _total_transfer_bytes{0};
    size_t _peak_device_bytes{0};
};

} // namespace einsums::compute_graph::passes
