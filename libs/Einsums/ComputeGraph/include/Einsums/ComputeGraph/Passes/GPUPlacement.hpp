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
 * @brief Decide which graph nodes should execute on a GPU.
 *
 * Two placement strategies:
 *
 * 1. **Size threshold** (default): A node is placed on GPU if its estimated_flops
 *    or estimated_bytes exceed configurable minimums.
 *
 * 2. **Cost model**: When estimated_flops is available, compares estimated CPU time
 *    vs GPU time + transfer overhead:
 *      - cpu_time  = flops / cpu_gflops
 *      - gpu_time  = flops / gpu_gflops + bytes / pcie_bandwidth
 *    Node is placed on GPU only when gpu_time < cpu_time.
 *
 * The cost model is used when estimated_flops > 0; otherwise falls back to
 * size thresholds. Budget-aware greedy placement ensures total device memory
 * is not exceeded.
 *
 * This pass does NOT insert transfer nodes, that is handled by TransferInsertion.
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
