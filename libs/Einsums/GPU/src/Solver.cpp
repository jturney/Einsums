//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/GPU/Error.hpp>
#include <Einsums/GPU/Solver.hpp>
#include <Einsums/GPU/Stream.hpp>

#if defined(EINSUMS_HAVE_CUDA)
#    include <cusolverDn.h>
#elif defined(EINSUMS_HAVE_HIP)
#    include <hipsolver/hipsolver.h>
#else
// Mock backend: delegate to CPU BLAS/LAPACK
#    include <Einsums/BLAS.hpp>
#    include <Einsums/BLAS/Types.hpp>
#    include <Einsums/BLASVendor/Vendor.hpp>
#endif

#if !defined(EINSUMS_HAVE_CUDA) && !defined(EINSUMS_HAVE_HIP)
namespace {
using int_t = ::einsums::blas::int_t;
}
#endif

#include <complex>
#include <vector>

namespace einsums::gpu::solver {

// ===========================================================================
// syev — Symmetric eigenvalue decomposition
// ===========================================================================

template <>
int syev<float>(char jobz, char uplo, int64_t n, float *A, int64_t lda, float *W) {
#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)A;
    (void)lda;
    (void)W;
    return -1;
#else
    return static_cast<int>(::einsums::blas::vendor::ssyevd(jobz, uplo, static_cast<int_t>(n), A, static_cast<int_t>(lda), W));
#endif
}

template <>
int syev<double>(char jobz, char uplo, int64_t n, double *A, int64_t lda, double *W) {
#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)A;
    (void)lda;
    (void)W;
    return -1;
#else
    return static_cast<int>(::einsums::blas::vendor::dsyevd(jobz, uplo, static_cast<int_t>(n), A, static_cast<int_t>(lda), W));
#endif
}

// ===========================================================================
// heev — Hermitian eigenvalue decomposition (complex)
// ===========================================================================

template <>
int heev<float>(char jobz, char uplo, int64_t n, std::complex<float> *A, int64_t lda, float *W) {
#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)A;
    (void)lda;
    (void)W;
    return -1;
#else
    std::complex<float> lwork_query;
    std::vector<float>  rwork(std::max(int64_t(1), 3 * n - 2));
    int info = ::einsums::blas::vendor::cheev(jobz, uplo, static_cast<int>(n), A, static_cast<int>(lda), W, &lwork_query, -1, rwork.data());
    int lwork = static_cast<int>(lwork_query.real());
    std::vector<std::complex<float>> work(lwork);
    info = ::einsums::blas::vendor::cheev(jobz, uplo, static_cast<int>(n), A, static_cast<int>(lda), W, work.data(), lwork, rwork.data());
    return info;
#endif
}

template <>
int heev<double>(char jobz, char uplo, int64_t n, std::complex<double> *A, int64_t lda, double *W) {
#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)A;
    (void)lda;
    (void)W;
    return -1;
#else
    std::complex<double> lwork_query;
    std::vector<double>  rwork(std::max(int64_t(1), 3 * n - 2));
    int info = ::einsums::blas::vendor::zheev(jobz, uplo, static_cast<int>(n), A, static_cast<int>(lda), W, &lwork_query, -1, rwork.data());
    int lwork = static_cast<int>(lwork_query.real());
    std::vector<std::complex<double>> work(lwork);
    info = ::einsums::blas::vendor::zheev(jobz, uplo, static_cast<int>(n), A, static_cast<int>(lda), W, work.data(), lwork, rwork.data());
    return info;
#endif
}

// ===========================================================================
// gesv — Linear solve AX = B
// ===========================================================================

template <>
int gesv<float>(int64_t n, int64_t nrhs, float *A, int64_t lda, int64_t *ipiv, float *B, int64_t ldb) {
#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)
    (void)n;
    (void)nrhs;
    (void)A;
    (void)lda;
    (void)ipiv;
    (void)B;
    (void)ldb;
    return -1;
#else
    // CPU gesv uses int* for ipiv, need to convert
    std::vector<int_t> ipiv32(n);
    int info = ::einsums::blas::vendor::sgesv(static_cast<int>(n), static_cast<int>(nrhs), A, static_cast<int>(lda), ipiv32.data(), B,
                                              static_cast<int>(ldb));
    for (int64_t i = 0; i < n; ++i)
        ipiv[i] = ipiv32[i];
    return info;
#endif
}

template <>
int gesv<double>(int64_t n, int64_t nrhs, double *A, int64_t lda, int64_t *ipiv, double *B, int64_t ldb) {
#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)
    (void)n;
    (void)nrhs;
    (void)A;
    (void)lda;
    (void)ipiv;
    (void)B;
    (void)ldb;
    return -1;
#else
    std::vector<int_t> ipiv32(n);
    int info = ::einsums::blas::vendor::dgesv(static_cast<int>(n), static_cast<int>(nrhs), A, static_cast<int>(lda), ipiv32.data(), B,
                                              static_cast<int>(ldb));
    for (int64_t i = 0; i < n; ++i)
        ipiv[i] = ipiv32[i];
    return info;
#endif
}

// ===========================================================================
// getrf — LU factorization
// ===========================================================================

template <>
int getrf<float>(int64_t m, int64_t n, float *A, int64_t lda, int64_t *ipiv) {
#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)
    (void)m;
    (void)n;
    (void)A;
    (void)lda;
    (void)ipiv;
    return -1;
#else
    std::vector<int_t> ipiv32(std::min(m, n));
    int info = ::einsums::blas::vendor::sgetrf(static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), ipiv32.data());
    for (int64_t i = 0; i < static_cast<int64_t>(ipiv32.size()); ++i)
        ipiv[i] = ipiv32[i];
    return info;
#endif
}

template <>
int getrf<double>(int64_t m, int64_t n, double *A, int64_t lda, int64_t *ipiv) {
#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)
    (void)m;
    (void)n;
    (void)A;
    (void)lda;
    (void)ipiv;
    return -1;
#else
    std::vector<int_t> ipiv32(std::min(m, n));
    int info = ::einsums::blas::vendor::dgetrf(static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), ipiv32.data());
    for (int64_t i = 0; i < static_cast<int64_t>(ipiv32.size()); ++i)
        ipiv[i] = ipiv32[i];
    return info;
#endif
}

// ===========================================================================
// getri — Inverse from LU factorization
// ===========================================================================

template <>
int getri<float>(int64_t n, float *A, int64_t lda, int64_t const *ipiv) {
#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)
    (void)n;
    (void)A;
    (void)lda;
    (void)ipiv;
    return -1;
#else
    std::vector<int_t> ipiv32(n);
    for (int64_t i = 0; i < n; ++i)
        ipiv32[i] = static_cast<int>(ipiv[i]);
    int info = ::einsums::blas::vendor::sgetri(static_cast<int>(n), A, static_cast<int>(lda), ipiv32.data());
    return info;
#endif
}

template <>
int getri<double>(int64_t n, double *A, int64_t lda, int64_t const *ipiv) {
#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)
    (void)n;
    (void)A;
    (void)lda;
    (void)ipiv;
    return -1;
#else
    std::vector<int_t> ipiv32(n);
    for (int64_t i = 0; i < n; ++i)
        ipiv32[i] = static_cast<int>(ipiv[i]);
    int info = ::einsums::blas::vendor::dgetri(static_cast<int>(n), A, static_cast<int>(lda), ipiv32.data());
    return info;
#endif
}

// ===========================================================================
// gesvd — Singular value decomposition
// ===========================================================================

template <>
int gesvd<float>(char jobu, char jobvt, int64_t m, int64_t n, float *A, int64_t lda, float *S, float *U, int64_t ldu, float *VT,
                 int64_t ldvt) {
#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)
    (void)jobu;
    (void)jobvt;
    (void)m;
    (void)n;
    (void)A;
    (void)lda;
    (void)S;
    (void)U;
    (void)ldu;
    (void)VT;
    (void)ldvt;
    return -1;
#else
    std::vector<float> superb(std::min(m, n));
    int info = ::einsums::blas::vendor::sgesvd(jobu, jobvt, static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), S, U,
                                               static_cast<int>(ldu), VT, static_cast<int>(ldvt), superb.data());
    return info;
#endif
}

template <>
int gesvd<double>(char jobu, char jobvt, int64_t m, int64_t n, double *A, int64_t lda, double *S, double *U, int64_t ldu, double *VT,
                  int64_t ldvt) {
#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)
    (void)jobu;
    (void)jobvt;
    (void)m;
    (void)n;
    (void)A;
    (void)lda;
    (void)S;
    (void)U;
    (void)ldu;
    (void)VT;
    (void)ldvt;
    return -1;
#else
    std::vector<double> superb(std::min(m, n));
    int info = ::einsums::blas::vendor::dgesvd(jobu, jobvt, static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), S, U,
                                               static_cast<int>(ldu), VT, static_cast<int>(ldvt), superb.data());
    return info;
#endif
}

} // namespace einsums::gpu::solver
