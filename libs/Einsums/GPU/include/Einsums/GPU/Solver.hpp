//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Types.hpp>

#include <complex>
#include <cstdint>

namespace einsums::gpu::solver {

// ===========================================================================
// GPU LAPACK operations.
// CUDA: cuSOLVER, HIP: hipSOLVER, Mock: delegates to CPU LAPACK
// ===========================================================================

// --- Symmetric/Hermitian eigenvalue decomposition ---

/// Symmetric eigenvalue: A → eigenvectors, W → eigenvalues.
/// jobz: 'V' = compute eigenvectors, 'N' = eigenvalues only.
/// uplo: 'U' = upper, 'L' = lower triangle.
template <typename T>
EINSUMS_EXPORT int syev(char jobz, char uplo, int64_t n, T *A, int64_t lda, T *W);

/// Hermitian eigenvalue (complex).
template <typename T>
EINSUMS_EXPORT int heev(char jobz, char uplo, int64_t n, std::complex<T> *A, int64_t lda, T *W);

// --- Linear solve: AX = B ---

template <typename T>
EINSUMS_EXPORT int gesv(int64_t n, int64_t nrhs, T *A, int64_t lda, int64_t *ipiv, T *B, int64_t ldb);

// --- LU factorization ---

template <typename T>
EINSUMS_EXPORT int getrf(int64_t m, int64_t n, T *A, int64_t lda, int64_t *ipiv);

// --- Inverse from LU ---

template <typename T>
EINSUMS_EXPORT int getri(int64_t n, T *A, int64_t lda, int64_t const *ipiv);

// --- SVD ---

template <typename T>
EINSUMS_EXPORT int gesvd(char jobu, char jobvt, int64_t m, int64_t n, T *A, int64_t lda,
                         std::conditional_t<std::is_same_v<T, std::complex<float>>, float,
                                            std::conditional_t<std::is_same_v<T, std::complex<double>>, double, T>> *S,
                         T *U, int64_t ldu, T *VT, int64_t ldvt);

} // namespace einsums::gpu::solver
