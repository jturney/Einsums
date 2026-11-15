//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file SymmetryDispatch.hpp
/// @brief Symmetry-aware dispatch: route rank-2 BLAS calls through the
/// specialized kernels (``symm``/``hemm``) when the operand carries a
/// matching ``SymmetryDescriptor``.
///
/// A ``false`` return means no match; the caller uses ``gemm``.

#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <cctype>

EINSUMS_NAMESPACE_BEGIN(linear_algebra::detail)

/// True iff ``desc`` describes a rank-2 symmetric matrix: a single
/// ``swap(0,1)`` generator with sign ``+1`` and no conjugation.
inline bool is_symmetric_rank2(SymmetryDescriptor const *desc) {
    if (!desc || desc->ops.size() != 1)
        return false;
    auto const &op = desc->ops[0];
    return op.permutation[0] == 1 && op.permutation[1] == 0 && op.sign == +1 && !op.conjugate;
}

/// True iff ``desc`` describes a rank-2 Hermitian matrix: a single
/// ``swap(0,1)`` generator with sign ``+1`` and conjugation.
inline bool is_hermitian_rank2(SymmetryDescriptor const *desc) {
    if (!desc || desc->ops.size() != 1)
        return false;
    auto const &op = desc->ops[0];
    return op.permutation[0] == 1 && op.permutation[1] == 0 && op.sign == +1 && op.conjugate;
}

/// If ``A`` or ``B`` is rank-2 symmetric (or Hermitian for complex types),
/// dispatch to ``blas::symm`` / ``blas::hemm`` and return ``true``. A
/// symmetric matrix is itself under ``'n'`` and ``'t'`` (Aᵀ = A), and under
/// ``'c'`` only when it is real: a complex symmetric matrix has Aᴴ = conj(A).
/// A Hermitian matrix is itself under ``'n'`` and ``'c'`` (Aᴴ = A); ``'t'``
/// on one has no free shortcut. Either way the OTHER operand must be ``'n'``,
/// because symm and hemm multiply it as stored. Anything else falls through.
///
/// Only handles the "all three tensors share element type ``T``" case;
/// mixed-precision / mixed-type gemms stay on the general path.
template <typename T>
bool try_symmetric_gemm(char transA, char transB, T alpha, einsums::detail::TensorImpl<T> const &A, SymmetryDescriptor const *desc_a,
                        einsums::detail::TensorImpl<T> const &B, SymmetryDescriptor const *desc_b, T beta,
                        einsums::detail::TensorImpl<T> *C) {
    char const tA = static_cast<char>(std::tolower(transA));
    char const tB = static_cast<char>(std::tolower(transB));

    // Eligibility per operand: the flag on the symmetric operand must leave it
    // unchanged, and the flag on the other must be 'n', since symm and hemm
    // take no op on it. A symmetric matrix is unchanged by 'n' and 't', and by
    // 'c' only when real: for a complex one Aᴴ = conj(A), a different matrix.
    // A Hermitian matrix is unchanged by 'n' and 'c'.
    auto sym_eligible = [](char t, SymmetryDescriptor const *d, char t_other) {
        return is_symmetric_rank2(d) && t_other == 'n' && (t == 'n' || t == 't' || (!einsums::IsComplexV<T> && t == 'c'));
    };
    auto her_eligible = [](char t, SymmetryDescriptor const *d, char t_other) {
        return is_hermitian_rank2(d) && t_other == 'n' && (t == 'n' || t == 'c');
    };

    // Both tensors must be gemmable (contiguous, no broadcasting, matching LD).
    if (!A.is_gemmable() || !B.is_gemmable() || !C->is_gemmable())
        return false;

    // symm: C = alpha * A * B + beta * C  (side='L', A is the symmetric matrix)
    //       C = alpha * B * A + beta * C  (side='R', A is the symmetric matrix,
    //                                      BLAS takes the symmetric one in slot A)
    //
    // All three tensors must share the same storage order; swap semantics
    // between row- and column-major mirror what impl_gemm_contiguous does.
    // For row-major we compute C^T = B^T·A^T in column-major BLAS and
    // exploit that the transpose of a symmetric matrix is the same matrix.
    bool const col_major = A.is_column_major() && B.is_column_major() && C->is_column_major();
    bool const row_major = A.is_row_major() && B.is_row_major() && C->is_row_major();
    if (!col_major && !row_major)
        return false;

    auto const m = C->dim(0);
    auto const n = C->dim(1);

    auto call_symm = [&](char side, T const *sym_ptr, int sym_lda, T const *oth_ptr, int oth_lda) {
        if (col_major) {
            blas::symm<T>(side, 'U', m, n, alpha, sym_ptr, sym_lda, oth_ptr, oth_lda, beta, C->data(), C->get_lda());
        } else {
            // Row-major: BLAS sees all matrices as column-major transposes.
            // C = alpha*A*B + beta*C  (sym on left)  ⟺  C^T = B^T*A + beta*C^T  →  symm(side='R', n, m, …)
            // C = alpha*A*B + beta*C  (sym on right) ⟺  C^T = B*A^T + beta*C^T  →  symm(side='L', n, m, …)
            char flipped = (side == 'L') ? 'R' : 'L';
            blas::symm<T>(flipped, 'U', n, m, alpha, sym_ptr, sym_lda, oth_ptr, oth_lda, beta, C->data(), C->get_lda());
        }
    };

    if (sym_eligible(tA, desc_a, tB)) {
        call_symm('L', A.data(), A.get_lda(), B.data(), B.get_lda());
        return true;
    }
    if (sym_eligible(tB, desc_b, tA)) {
        call_symm('R', B.data(), B.get_lda(), A.data(), A.get_lda());
        return true;
    }

    if constexpr (einsums::IsComplexV<T>) {
        // hemm: Hermitian matrix where A^T ≠ A, so the row-major swap needs
        // extra thought. For side='L', row-major C = A·B becomes column-major
        // C^T = B^T·A^T. A^T = conj(A) for Hermitian A, not A itself. BLAS
        // has no direct "conj-Hermitian" kernel, so we only fire hemm on the
        // column-major layout. Row-major Hermitian falls through to gemm.
        if (!col_major)
            return false;

        auto call_hemm = [&](char side, T const *sym_ptr, int sym_lda, T const *oth_ptr, int oth_lda) {
            blas::hemm<T>(side, 'U', m, n, alpha, sym_ptr, sym_lda, oth_ptr, oth_lda, beta, C->data(), C->get_lda());
        };
        if (her_eligible(tA, desc_a, tB)) {
            call_hemm('L', A.data(), A.get_lda(), B.data(), B.get_lda());
            return true;
        }
        if (her_eligible(tB, desc_b, tA)) {
            call_hemm('R', B.data(), B.get_lda(), A.data(), A.get_lda());
            return true;
        }
    }

    return false;
}

EINSUMS_NAMESPACE_END(linear_algebra::detail)
