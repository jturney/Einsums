//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile.hpp>

#include <array>

#include "Common.hpp"

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

// Batch GEMM: perform batch_count independent GEMMs.
// All batches share the same transa, transb, m, n, k, alpha, beta, lda, ldb, ldc.
// Only the data pointers differ (passed as arrays).
//
// This is a fallback implementation that calls the regular Fortran GEMM in a
// parallel loop. If a vendor provides a native batch GEMM (MKL Fortran
// dgemm_batch, or OpenBLAS cblas_dgemm_batch), a vendor-specific override
// can be added here behind a CMake config define.

extern "C" {
extern void FC_GLOBAL(sgemm, SGEMM)(char *, char *, int_t *, int_t *, int_t *, float *, float const *, int_t *, float const *, int_t *,
                                    float *, float *, int_t *);
extern void FC_GLOBAL(dgemm, DGEMM)(char *, char *, int_t *, int_t *, int_t *, double *, double const *, int_t *, double const *, int_t *,
                                    double *, double *, int_t *);
extern void FC_GLOBAL(cgemm, CGEMM)(char *, char *, int_t *, int_t *, int_t *, std::complex<float> *, std::complex<float> const *, int_t *,
                                    std::complex<float> const *, int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zgemm, ZGEMM)(char *, char *, int_t *, int_t *, int_t *, std::complex<double> *, std::complex<double> const *,
                                    int_t *, std::complex<double> const *, int_t *, std::complex<double> *, std::complex<double> *,
                                    int_t *);
}

namespace {

/// Largest dimension that the inline kernel below will take from the vendor GEMM.
constexpr int_t small_gemm_dim = 16;

template <typename T>
constexpr bool is_complex_v = false;
template <typename U>
constexpr bool is_complex_v<std::complex<U>> = true;

template <typename T>
inline T conj_if(T v, bool conj) {
    if constexpr (is_complex_v<T>) {
        return conj ? std::conj(v) : v;
    } else {
        return v;
    }
}

/// One tiny column-major GEMM, written so the innermost loop walks the unit-stride index.
template <typename T>
void small_gemm(char transa, char transb, int_t m, int_t n, int_t k, T alpha, T const *a, int_t lda, T const *b, int_t ldb, T beta, T *c,
                int_t ldc) {
    bool const ta = transa != 'N' && transa != 'n';
    bool const tb = transb != 'N' && transb != 'n';
    bool const ca = transa == 'C' || transa == 'c';
    bool const cb = transb == 'C' || transb == 'c';

    std::array<T, small_gemm_dim * small_gemm_dim> acc;
    for (int_t j = 0; j < n; j++) {
        T *cj = acc.data() + j * m;
        for (int_t i = 0; i < m; i++) {
            cj[i] = T{};
        }
        for (int_t p = 0; p < k; p++) {
            T const bp = conj_if(tb ? b[j + p * ldb] : b[p + j * ldb], cb);
            if (ta) {
                for (int_t i = 0; i < m; i++) {
                    cj[i] += conj_if(a[p + i * lda], ca) * bp;
                }
            } else {
                T const *ap = a + p * lda;
                for (int_t i = 0; i < m; i++) {
                    cj[i] += ap[i] * bp;
                }
            }
        }
    }
    // beta == 0 must not read C at all: BLAS lets the caller pass uninitialized memory.
    if (beta == T{}) {
        for (int_t j = 0; j < n; j++) {
            for (int_t i = 0; i < m; i++) {
                c[i + j * ldc] = alpha * acc[i + j * m];
            }
        }
    } else {
        for (int_t j = 0; j < n; j++) {
            for (int_t i = 0; i < m; i++) {
                c[i + j * ldc] = alpha * acc[i + j * m] + beta * c[i + j * ldc];
            }
        }
    }
}

/// Should this batch run on the inline kernel rather than on the vendor GEMM?
///
/// Only when the loop below will actually be concurrent. A vendor GEMM beats the
/// kernel on a single thread at every size - 0.11 us against 0.22 for 9x9 doubles
/// here - because it is a better kernel. What it does not survive is being called
/// concurrently: OpenBLAS serializes inside each call, so the same 4000-element
/// 9x9 batch costs 0.44 ms on one thread and 1.04 ms on ten, while the kernel below
/// shares nothing and goes 0.85 -> 0.21. The crossover against the vendor rises with
/// the thread count (dim 10 at four threads, dim 16 at ten), so the cutoff is set
/// where it holds for a wide team and left alone for a narrow one.
inline bool use_small_gemm(int_t m, int_t n, int_t k) {
#ifdef _OPENMP
    return m <= small_gemm_dim && n <= small_gemm_dim && k <= small_gemm_dim && omp_get_max_threads() > 1;
#else
    return false;
#endif
}

} // namespace

#ifdef _OPENMP
#    define EINSUMS_SMALL_GEMM_BATCH(TYPE)                                                                                                 \
        if (use_small_gemm(m, n, k)) {                                                                                                     \
            _Pragma(                                                                                                                       \
                "omp parallel for schedule(static) firstprivate(transa, transb, m, n, k, alpha, lda, ldb, beta, ldc)") for (int_t i = 0;   \
                                                                                                                            i <            \
                                                                                                                            batch_count;   \
                                                                                                                            i++) {         \
                small_gemm<TYPE>(transa, transb, m, n, k, alpha, a_array[i], lda, b_array[i], ldb, beta, c_array[i], ldc);                 \
            }                                                                                                                              \
            return;                                                                                                                        \
        }
#else
#    define EINSUMS_SMALL_GEMM_BATCH(TYPE)
#endif

// LAPACK's gemm takes its scalar arguments by pointer (Fortran ABI), so we pass
// addresses of the function-locals. Without ``firstprivate``, every OMP worker
// reads those addresses out of the caller's stack frame, which is fine while the
// parallel region is live (workers only read; gemm never writes its inputs)
// but TSan can't see libgomp's barrier and so reports a race against any
// later write to the same stack region (e.g. the next fmt formatting in the
// caller). ``firstprivate`` gives each worker its own private copies and
// unique addresses, eliminating both the (real but harmless) shared-stack
// read pattern and the (libgomp-invisibility) TSan false positive.

void sgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const **a_array, int_t lda, float const **b_array,
                 int_t ldb, float beta, float **c_array, int_t ldc, int_t batch_count) {
    LabeledSection0();
    if (batch_count <= 0 || m == 0 || n == 0)
        return;

    EINSUMS_SMALL_GEMM_BATCH(float)

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic) firstprivate(transa, transb, m, n, k, alpha, lda, ldb, beta, ldc)
#endif
    for (int_t i = 0; i < batch_count; i++) {
        FC_GLOBAL(sgemm, SGEMM)
        (&transa, &transb, &m, &n, &k, &alpha, a_array[i], &lda, b_array[i], &ldb, &beta, c_array[i], &ldc);
    }
}

void dgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const **a_array, int_t lda,
                 double const **b_array, int_t ldb, double beta, double **c_array, int_t ldc, int_t batch_count) {
    LabeledSection0();
    if (batch_count <= 0 || m == 0 || n == 0)
        return;

    EINSUMS_SMALL_GEMM_BATCH(double)

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic) firstprivate(transa, transb, m, n, k, alpha, lda, ldb, beta, ldc)
#endif
    for (int_t i = 0; i < batch_count; i++) {
        FC_GLOBAL(dgemm, DGEMM)
        (&transa, &transb, &m, &n, &k, &alpha, a_array[i], &lda, b_array[i], &ldb, &beta, c_array[i], &ldc);
    }
}

void cgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const **a_array,
                 int_t lda, std::complex<float> const **b_array, int_t ldb, std::complex<float> beta, std::complex<float> **c_array,
                 int_t ldc, int_t batch_count) {
    LabeledSection0();
    if (batch_count <= 0 || m == 0 || n == 0)
        return;

    EINSUMS_SMALL_GEMM_BATCH(std::complex<float>)

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic) firstprivate(transa, transb, m, n, k, alpha, lda, ldb, beta, ldc)
#endif
    for (int_t i = 0; i < batch_count; i++) {
        FC_GLOBAL(cgemm, CGEMM)
        (&transa, &transb, &m, &n, &k, &alpha, a_array[i], &lda, b_array[i], &ldb, &beta, c_array[i], &ldc);
    }
}

void zgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const **a_array,
                 int_t lda, std::complex<double> const **b_array, int_t ldb, std::complex<double> beta, std::complex<double> **c_array,
                 int_t ldc, int_t batch_count) {
    LabeledSection0();
    if (batch_count <= 0 || m == 0 || n == 0)
        return;

    EINSUMS_SMALL_GEMM_BATCH(std::complex<double>)

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic) firstprivate(transa, transb, m, n, k, alpha, lda, ldb, beta, ldc)
#endif
    for (int_t i = 0; i < batch_count; i++) {
        FC_GLOBAL(zgemm, ZGEMM)
        (&transa, &transb, &m, &n, &k, &alpha, a_array[i], &lda, b_array[i], &ldb, &beta, c_array[i], &ldc);
    }
}

EINSUMS_NAMESPACE_END(blas::vendor)
