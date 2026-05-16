//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Permute-into-einsum fusion.
 *
 * Absorbs an axis-reordering Permute node into the subscript of the
 * Einsum that reads it. Eliminates one tensor-shaped data copy (and,
 * on GPU, one kernel launch + round-trip) per match.
 *
 * @par Pattern:
 * @code
 * // Before:
 * cg::permute("ji <- ij", &A_T, A);      // physical transpose
 * cg::einsum("ji;jk->ik", &C, A_T, B);   // GEMM on transposed A
 *
 * // After:
 * cg::einsum("ij;jk->ik", &C, A, B);     // subscript rewritten, A read directly
 * @endcode
 *
 * @par Safety conditions (all must hold):
 *  - The Permute's output has exactly one consumer (the Einsum).
 *  - The Permute is a pure axis reordering: `alpha == 1 && beta == 0`
 *    and `c_indices` is a duplicate-free permutation of `a_indices`.
 *  - The consumer is an Einsum whose `EinsumDescriptor::indices`
 *    shared state is populated (the mutable-indices infrastructure).
 *
 * @par Mechanism:
 * The rewrite mutates @ref EinsumIndices in place (captured by the
 * executor via shared_ptr), so the change takes effect on the next
 * `graph.execute()` without rebuilding the executor lambda. The
 * snapshot in @ref EinsumDescriptor::spec is also updated so analysis
 * passes see consistent state. The Permute node is then removed.
 */
class EINSUMS_EXPORT PermuteFusion : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "PermuteFusion"; }
    bool                      run(Graph &graph) override;

    /// Number of Permute→Einsum/Gemm pairs detected this run (before safety filtering).
    [[nodiscard]] size_t num_candidates() const { return _num_candidates; }

    /// Number of candidates that passed safety checks and were actually rewritten.
    [[nodiscard]] size_t num_rewrites() const { return _num_rewrites; }

  private:
    size_t _num_candidates{0};
    size_t _num_rewrites{0};
};

} // namespace einsums::compute_graph::passes
