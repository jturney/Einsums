//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file Orthogonalization.hpp
 * @brief Orthogonalization blueprints for computation graphs.
 *
 * Provides common orthogonalization procedures used in quantum chemistry:
 * - Symmetric orthogonalization: X = S^{-1/2}
 * - Canonical orthogonalization: X = U * s^{-1/2} (with linear dependency removal)
 */

#include <Einsums/ComputeGraph/Operations.hpp>

#include <cmath>

namespace einsums::compute_graph::blueprints {

/**
 * @brief Symmetric orthogonalization: X = S^{-1/2}.
 *
 * Computes the orthogonalization matrix from an overlap matrix S using
 * the symmetric (Lowdin) procedure:
 *   1. Diagonalize S: S = U * diag(s) * U^T
 *   2. Compute s^{-1/2} for each eigenvalue
 *   3. X = U * diag(s^{-1/2}) * U^T
 *
 * @tparam MatType Matrix tensor type.
 * @param[out] X The orthogonalization matrix (same dimensions as S).
 * @param[in] S The overlap matrix (must be symmetric positive definite).
 *
 * @note This blueprint always executes eagerly (even during capture) because
 *       it uses syev which is a return-value operation. The result is then
 *       used by subsequent captured operations.
 *
 * @code
 * cg::blueprints::orthogonalize(&X, S);
 * // X now contains S^{-1/2}
 * @endcode
 */
template <MatrixConcept MatType>
void orthogonalize(MatType *X, MatType const &S) {
    using T        = typename MatType::ValueType;
    size_t const n = S.dim(0);

    // Always execute eagerly — syev is a return-value operation
    auto [U, s] = linear_algebra::syev(S);

    // s^{-1/2}
    tensor_algebra::element_transform(&s, [](T val) { return T{1} / std::sqrt(val); });

    // U_scaled = U * diag(s^{-1/2}): scale columns of a copy
    auto U_scaled = RemoveViewT<MatType>(U); // Deep copy
    for (size_t col = 0; col < n; col++) {
        linear_algebra::scale_column(col, s(col), &U_scaled);
    }

    // X = U_scaled * U^T = U * diag(s^{-1/2}) * U^T = S^{-1/2}
    linear_algebra::gemm<false, true>(T{1}, U_scaled, U, T{0}, X);
}

/**
 * @brief Canonical orthogonalization: X = U * diag(s^{-1/2}), with linear dependency removal.
 *
 * Similar to symmetric orthogonalization but removes eigenvectors
 * corresponding to eigenvalues below a threshold (near-linear dependencies).
 * The result X may have fewer columns than rows if dependencies are removed.
 *
 * For simplicity, this version still produces an n×n matrix but zeros out
 * columns corresponding to removed eigenvalues. A more sophisticated version
 * would return a rectangular matrix.
 *
 * @tparam MatType Matrix tensor type.
 * @param[out] X The orthogonalization matrix.
 * @param[in] S The overlap matrix.
 * @param[in] threshold Eigenvalues below this are considered linearly dependent (default 1e-6).
 * @return Number of eigenvalues removed.
 *
 * @code
 * size_t removed = cg::blueprints::canonical_orthogonalize(&X, S, 1e-6);
 * println("Removed {} linearly dependent functions", removed);
 * @endcode
 */
template <MatrixConcept MatType>
size_t canonical_orthogonalize(MatType *X, MatType const &S, typename MatType::ValueType threshold = typename MatType::ValueType{1e-6}) {
    using T        = typename MatType::ValueType;
    size_t const n = S.dim(0);

    // Always execute eagerly
    auto [U, s] = linear_algebra::syev(S);

    // Count and remove linear dependencies
    size_t removed = 0;
    for (size_t col = 0; col < n; col++) {
        if (s(col) < threshold) {
            s(col) = T{0}; // Zero out this eigenvalue
            removed++;
        } else {
            s(col) = T{1} / std::sqrt(s(col));
        }
    }

    // U_scaled = U * diag(s^{-1/2}) (or zero for removed)
    auto U_scaled = RemoveViewT<MatType>(U);
    for (size_t col = 0; col < n; col++) {
        linear_algebra::scale_column(col, s(col), &U_scaled);
    }

    // X = U_scaled * U^T
    linear_algebra::gemm<false, true>(T{1}, U_scaled, U, T{0}, X);

    return removed;
}

} // namespace einsums::compute_graph::blueprints
