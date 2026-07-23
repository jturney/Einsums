//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/HardwareProfile.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief GPU-offload placement pass: flags graph nodes to execute on a GPU.
 *
 * @note GPU offload is a work in progress. This pass runs only on GPU-enabled (or mock)
 *       builds, and the transfer/execution backend it targets is not yet complete.
 *
 * Walks the whole graph tree (including loop bodies and conditional branches, so a hot
 * GEMM inside an SCF loop is a candidate) and marks profitable BLAS/LAPACK ops and
 * einsums with `Target::GPU`, saving each node's CPU executor as its `cpu_fallback`.
 * It does NOT insert host/device copies — @ref TransferInsertion does that afterward.
 *
 * Two placement strategies decide profitability per node:
 *
 * 1. **Cost model** (used when `estimated_flops > 0`): compares CPU time against GPU
 *    time plus transfer and launch overhead —
 *      - `cpu_time = flops / cpu_gflops`
 *      - `gpu_time = flops / gpu_gflops + bytes / pcie_bandwidth + launch_overhead`
 *    and places the node only when `gpu_time < cpu_time`. Constructing with a
 *    @ref HardwareProfile replaces the hardcoded throughput/bandwidth constants.
 *
 * 2. **Size threshold** (fallback when `estimated_flops == 0`): places the node only
 *    when its estimated memory traffic reaches `min_bytes`.
 *
 * Profitable candidates are then placed greedily, largest-byte-footprint first, until
 * `gpu::available_device_memory()` is exhausted; nodes that no longer fit stay on CPU.
 *
 * In `create_default()` this is the first pass inside the GPU-enabled block
 * (`if constexpr (gpu::has_gpu || gpu::is_mock)`), constructed with the shared
 * `HardwareProfile`; the whole block is compiled out on CPU-only builds.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("gpu_placement");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);     // a large GEMM
 * }
 * graph.apply(cg::PassManager::create_default());          // GPUPlacement runs iff a GPU/mock backend is compiled in
 * graph.execute();
 * // When a GPU backend is present and the cost model favors it, the einsum
 * // node's target is now Target::GPU (with its CPU executor kept as fallback).
 * @endcode
 *
 * @par Example (Python)
 * No standalone Python entry point: this pass is not individually exposed and only runs
 * inside the distributed/GPU pipeline.
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("gpu_placement")
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", C, A, B)
 * g.apply(cg.default_pass_manager())                       # GPUPlacement participates only on a GPU-enabled build
 * @endcode
 *
 * @par Limitations
 * - Compiled out entirely on CPU-only builds; a no-op when `--einsums:disable-gpu` is set,
 *   so on the CI default (no GPU, no mock) the pass never fires.
 * - The **MPS** backend supports only `float32`; `float64`/complex nodes are rejected by the
 *   dtype guard and left on CPU. CUDA/HIP/mock accept `float32`/`float64`/complex64/complex128.
 * - Budget placement is greedy by descending byte footprint, not globally optimal: a node
 *   larger than the remaining budget is skipped even if reordering could have fit it.
 * - The budget accounting double-counts residency — a tensor shared by several placed nodes
 *   is charged its bytes per node, so the effective budget is conservative.
 * - The default throughput/PCIe/launch constants are coarse; without a `HardwareProfile` they
 *   are placeholders rather than measured device numbers.
 *
 * @par Future improvements
 * - Multi-GPU placement (a single device budget is assumed today).
 * - Credit already-resident tensors in the budget instead of re-charging shared operands.
 * - Feed measured device profiles into the cost-model constants so size-threshold fallback is
 *   needed less often.
 */
class EINSUMS_EXPORT GPUPlacement : public OptimizerPass {
  public:
    /**
     * @param[in] min_flops  Minimum estimated FLOPs for a node to be placed on GPU (size-threshold mode).
     * @param[in] min_bytes  Minimum estimated memory traffic (bytes) for GPU placement.
     */
    explicit GPUPlacement(size_t min_flops = 100000, size_t min_bytes = 65536);

    /**
     * @brief Construct with a HardwareProfile for cost model parameters.
     *
     * The profile's CPU and GPU device data replace the hardcoded cost
     * model parameters (cpu_throughput_gflops, gpu_throughput_gflops, etc.).
     */
    explicit GPUPlacement(HardwareProfile const &profile, size_t min_flops = 100000, size_t min_bytes = 65536);

    [[nodiscard]] std::string name() const override { return "GPUPlacement"; }
    bool                      run(Graph &graph) override;

    /// Number of nodes placed on GPU in the last run.
    [[nodiscard]] size_t num_placed() const { return _num_placed; }

    /// Cost model parameters. When constructed with a HardwareProfile,
    /// these are populated from the profile. Otherwise, sensible defaults.
    double cpu_throughput_gflops{50.0};   ///< Estimated CPU throughput (GFLOP/s)
    double gpu_throughput_gflops{5000.0}; ///< Estimated GPU throughput (GFLOP/s)
    double pcie_bandwidth_gbs{12.0};      ///< PCIe bandwidth (GB/s) for transfer overhead
    double gpu_launch_overhead_us{10.0};  ///< GPU kernel launch overhead (microseconds)

  private:
    size_t _min_flops;
    size_t _min_bytes;
    size_t _num_placed{0};
};

} // namespace einsums::compute_graph::passes
