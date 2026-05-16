//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace einsums::blas::vendor {

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

void sgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const **a_array, int_t lda, float const **b_array,
                 int_t ldb, float beta, float **c_array, int_t ldc, int_t batch_count) {
    LabeledSection0();
    if (batch_count <= 0 || m == 0 || n == 0)
        return;

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic)
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

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic)
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

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic)
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

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic)
#endif
    for (int_t i = 0; i < batch_count; i++) {
        FC_GLOBAL(zgemm, ZGEMM)
        (&transa, &transb, &m, &n, &k, &alpha, a_array[i], &lda, b_array[i], &ldb, &beta, c_array[i], &ldc);
    }
}

} // namespace einsums::blas::vendor
