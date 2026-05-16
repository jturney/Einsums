//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Optimal matrix chain parenthesization pass (analysis only).
 *
 * Detects chains of matrix multiplications in the graph — sequences of Einsum
 * nodes with GEMM pattern (C[i,j] = A[i,k] * B[k,j]) where each node's output
 * feeds the next node as input. Applies the classical O(n³) dynamic programming
 * algorithm to find the optimal multiplication order.
 *
 * **Why this matters:**
 *
 * For a chain A(100×1) × B(1×100) × C(100×1):
 * - Left-to-right: (A×B)×C → 20,000 + 20,000 = 40,000 FLOPs
 * - Optimal: A×(B×C) → 200 + 200 = 400 FLOPs (100× improvement!)
 *
 * **What it does:**
 * - Identifies GEMM-pattern einsum chains in the graph
 * - Extracts M, K, N dimensions from tensor handles
 * - Runs the matrix chain DP algorithm
 * - Logs the optimal ordering and FLOP savings
 * - Reports original_flops() vs optimal_flops()
 *
 * **What it does NOT do:**
 * - Does NOT restructure the graph (that would require creating new intermediate
 *   tensors with type information that is erased at capture time)
 * - The user should use the recommendations to restructure their code
 *
 * @par Example
 * @code
 * auto [modified, chain] = graph.apply<cg::passes::ChainParenthesization>();
 *
 * println("Original FLOPs: {}", chain.original_flops());
 * println("Optimal FLOPs:  {}", chain.optimal_flops());
 * println("Speedup: {:.1f}x",
 *     double(chain.original_flops()) / double(chain.optimal_flops()));
 * @endcode
 */
class EINSUMS_EXPORT ChainParenthesization : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "ChainParenthesization"; }

    /**
     * @brief Run the chain parenthesization analysis.
     * @param[in,out] graph The graph to analyze (not modified).
     * @return Always returns false (analysis only).
     */
    bool run(Graph &graph) override;

    /**
     * @brief Total FLOPs with the original (left-to-right) evaluation order.
     * @return FLOP count, or 0 if no chains were detected.
     */
    [[nodiscard]] size_t original_flops() const { return _original_flops; }

    /**
     * @brief Total FLOPs with the optimal parenthesization.
     * @return FLOP count, or 0 if no chains were detected.
     */
    [[nodiscard]] size_t optimal_flops() const { return _optimal_flops; }

  private:
    size_t _original_flops{0};
    size_t _optimal_flops{0};
};

} // namespace einsums::compute_graph::passes
