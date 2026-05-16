//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/HardwareProfile.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <string>
#include <vector>

namespace einsums::compute_graph::passes {

/**
 * @brief Multi-objective contraction planning pass.
 *
 * Detects chains of GEMM-pattern einsum nodes and computes the optimal
 * contraction order using a cost model that considers:
 *   - FLOPs (compute time scaled by shape-dependent GEMM efficiency)
 *   - Memory traffic (read/write bandwidth to main memory)
 *   - Kernel overhead (BLAS call and allocation costs)
 *   - GEMM shape efficiency (small/skinny GEMMs are slower per FLOP)
 *
 * Unlike ChainParenthesization (analysis-only), this pass actually
 * restructures the graph: it creates intermediate tensors and rewrites
 * the contraction sequence to minimize estimated wall-clock time.
 *
 * The cost model uses a HardwareProfile that can be auto-detected,
 * loaded from a JSON calibration file, or provided programmatically.
 *
 * @par Example
 * @code
 * // With default hardware detection:
 * pm.add<cg::passes::ContractionPlanning>();
 *
 * // With calibrated profile:
 * pm.add<cg::passes::ContractionPlanning>(
 *     HardwareProfile::load_json("my_hardware.json"));
 *
 * // After running:
 * auto [modified, cp] = graph.apply<cg::passes::ContractionPlanning>();
 * for (auto const &r : cp.chain_reports()) {
 *     println("Chain of {} GEMMs: {:.1f}x speedup", r.chain_length, r.speedup);
 * }
 * @endcode
 */
class EINSUMS_EXPORT ContractionPlanning : public OptimizerPass {
  public:
    /// Construct with auto-detected hardware profile.
    ContractionPlanning();

    /// Construct with a specific hardware profile.
    explicit ContractionPlanning(HardwareProfile profile);

    [[nodiscard]] std::string name() const override { return "ContractionPlanning"; }
    bool                      run(Graph &graph) override;

    // ── Report ─────────────────────────────────────────────────────────────

    /// Per-chain report from the last run.
    struct ChainReport {
        size_t              chain_length{0};
        std::vector<size_t> dimensions; ///< p[0..n] dimension array
        double              original_time_us{0};
        double              optimal_time_us{0};
        double              speedup{1.0};
        size_t              intermediates_created{0};
        double              comm_cost_us{0};        ///< Estimated communication cost (allreduce etc.)
        bool                has_distributed{false}; ///< True if chain involves distributed tensors
    };

    [[nodiscard]] std::vector<ChainReport> const &chain_reports() const { return _reports; }
    [[nodiscard]] size_t                          chains_restructured() const { return _chains_restructured; }
    [[nodiscard]] size_t                          intermediates_created() const { return _intermediates_created; }

  private:
    HardwareProfile          _profile;
    std::vector<ChainReport> _reports;
    size_t                   _chains_restructured{0};
    size_t                   _intermediates_created{0};
};

} // namespace einsums::compute_graph::passes
