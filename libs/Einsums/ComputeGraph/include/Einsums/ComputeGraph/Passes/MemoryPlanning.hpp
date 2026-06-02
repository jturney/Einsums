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
 * @brief Liveness analysis and memory planning pass (analysis only).
 *
 * Analyzes the graph to determine tensor liveness intervals (from first use
 * to last use) and computes memory statistics for both host and device:
 * - **total_memory()**: Sum of all tensor sizes (upper bound if all were live simultaneously)
 * - **peak_memory()**: Maximum sum of simultaneously live tensors at any execution point
 * - **device_total_memory()**: Sum of all device-resident tensor sizes
 * - **device_peak_memory()**: Peak simultaneously live device tensors
 * - **device_reuse_savings()**: Bytes saveable by reusing non-overlapping device allocations
 *
 * @note This pass does NOT modify the graph — it is analysis only.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT MemoryPlanning : public OptimizerPass {
  public:
    APIARY_EXPOSE MemoryPlanning() = default;

    [[nodiscard]] std::string name() const override { return "MemoryPlanning"; }
    bool                      run(Graph &graph) override;
    void                      print_report(std::ostream &os) const;

    APIARY_EXPOSE APIARY_GETTER("total_memory") [[nodiscard]] size_t total_memory() const { return _total_memory; }
    APIARY_EXPOSE APIARY_GETTER("peak_memory") [[nodiscard]] size_t peak_memory() const { return _peak_memory; }

    /// Device memory: total across all device-resident tensors.
    APIARY_EXPOSE APIARY_GETTER("device_total_memory") [[nodiscard]] size_t device_total_memory() const { return _device_total_memory; }

    /// Device memory: peak simultaneously live.
    [[nodiscard]] size_t device_peak_memory() const { return _device_peak_memory; }

    /// Bytes saveable by reusing non-overlapping device allocations.
    [[nodiscard]] size_t device_reuse_savings() const {
        return _device_total_memory > _device_peak_memory ? _device_total_memory - _device_peak_memory : 0;
    }

  private:
    size_t _total_memory{0};
    size_t _peak_memory{0};
    size_t _device_total_memory{0};
    size_t _device_peak_memory{0};
};

} // namespace einsums::compute_graph::passes
