//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Types.hpp>

#include <complex>
#include <cstdint>

namespace einsums::gpu::blas {

// ===========================================================================
// Standard precision GEMM — matches CPU BLAS signatures.
// CUDA: cuBLAS, HIP: hipBLAS, Mock: delegates to CPU einsums::blas::vendor
// ===========================================================================

EINSUMS_EXPORT void sgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float const *a, int64_t lda,
                          float const *b, int64_t ldb, float beta, float *c, int64_t ldc);

EINSUMS_EXPORT void dgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double const *a, int64_t lda,
                          double const *b, int64_t ldb, double beta, double *c, int64_t ldc);

EINSUMS_EXPORT void cgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, std::complex<float> alpha,
                          std::complex<float> const *a, int64_t lda, std::complex<float> const *b, int64_t ldb, std::complex<float> beta,
                          std::complex<float> *c, int64_t ldc);

EINSUMS_EXPORT void zgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, std::complex<double> alpha,
                          std::complex<double> const *a, int64_t lda, std::complex<double> const *b, int64_t ldb, std::complex<double> beta,
                          std::complex<double> *c, int64_t ldc);

/// Template wrapper dispatching to the type-specific GEMM.
template <typename T>
EINSUMS_EXPORT void gemm(char transa, char transb, int64_t m, int64_t n, int64_t k, T alpha, T const *a, int64_t lda, T const *b,
                         int64_t ldb, T beta, T *c, int64_t ldc);

// ===========================================================================
// Strided-batched GEMM — N independent 2D GEMMs with matrices stored
// contiguously in memory with a uniform stride between them. Maps to
// cublasDgemmStridedBatched / hipblasDgemmStridedBatched on discrete
// GPUs; on CPU/mock builds, falls through to a loop over the CPU
// gemm_batch wrapper with pointers computed from base + i*stride.
//
// Compared to the pointer-array form (not yet wrapped here), strided
// batching avoids per-call pointer-array construction and allows the
// GPU runtime to schedule the whole batch as one work unit.
// ===========================================================================

EINSUMS_EXPORT void sgemm_strided_batched(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float const *a,
                                          int64_t lda, int64_t stride_a, float const *b, int64_t ldb, int64_t stride_b, float beta,
                                          float *c, int64_t ldc, int64_t stride_c, int64_t batch_count);

EINSUMS_EXPORT void dgemm_strided_batched(char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double const *a,
                                          int64_t lda, int64_t stride_a, double const *b, int64_t ldb, int64_t stride_b, double beta,
                                          double *c, int64_t ldc, int64_t stride_c, int64_t batch_count);

EINSUMS_EXPORT void cgemm_strided_batched(char transa, char transb, int64_t m, int64_t n, int64_t k, std::complex<float> alpha,
                                          std::complex<float> const *a, int64_t lda, int64_t stride_a, std::complex<float> const *b,
                                          int64_t ldb, int64_t stride_b, std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                                          int64_t stride_c, int64_t batch_count);

EINSUMS_EXPORT void zgemm_strided_batched(char transa, char transb, int64_t m, int64_t n, int64_t k, std::complex<double> alpha,
                                          std::complex<double> const *a, int64_t lda, int64_t stride_a, std::complex<double> const *b,
                                          int64_t ldb, int64_t stride_b, std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                                          int64_t stride_c, int64_t batch_count);

/// Template wrapper dispatching to the type-specific strided-batched GEMM.
template <typename T>
EINSUMS_EXPORT void gemm_strided_batched(char transa, char transb, int64_t m, int64_t n, int64_t k, T alpha, T const *a, int64_t lda,
                                         int64_t stride_a, T const *b, int64_t ldb, int64_t stride_b, T beta, T *c, int64_t ldc,
                                         int64_t stride_c, int64_t batch_count);

// ===========================================================================
// Standard precision GEMV — matches CPU BLAS signatures.
// ===========================================================================

EINSUMS_EXPORT void sgemv(char trans, int64_t m, int64_t n, float alpha, float const *a, int64_t lda, float const *x, int64_t incx,
                          float beta, float *y, int64_t incy);

EINSUMS_EXPORT void dgemv(char trans, int64_t m, int64_t n, double alpha, double const *a, int64_t lda, double const *x, int64_t incx,
                          double beta, double *y, int64_t incy);

/// Template wrapper dispatching to the type-specific GEMV.
template <typename T>
EINSUMS_EXPORT void gemv(char trans, int64_t m, int64_t n, T alpha, T const *a, int64_t lda, T const *x, int64_t incx, T beta, T *y,
                         int64_t incy);

// ===========================================================================
// BLAS Level 1 — element-wise operations on device memory.
// These avoid unnecessary D2H/H2D round-trips when data is already on GPU.
// ===========================================================================

/**
 * @brief Scale a vector on device memory: x = alpha * x.
 * @param[in] n      Number of elements.
 * @param[in] alpha  Scale factor.
 * @param[in,out] x  Device pointer to the vector.
 * @param[in] incx   Stride between consecutive elements of x.
 */
template <typename T>
EINSUMS_EXPORT void scal(int64_t n, T alpha, T *x, int64_t incx);

/**
 * @brief Axpy on device memory: y = alpha * x + y.
 * @param[in] n      Number of elements.
 * @param[in] alpha  Scale factor for x.
 * @param[in] x      Device pointer to source vector (read-only).
 * @param[in] incx   Stride for x.
 * @param[in,out] y  Device pointer to destination vector (accumulated).
 * @param[in] incy   Stride for y.
 */
template <typename T>
EINSUMS_EXPORT void axpy(int64_t n, T alpha, T const *x, int64_t incx, T *y, int64_t incy);

/**
 * @brief Axpby on device memory: y = alpha * x + beta * y.
 * @param[in] n      Number of elements.
 * @param[in] alpha  Scale factor for x.
 * @param[in] x      Device pointer to source vector.
 * @param[in] incx   Stride for x.
 * @param[in] beta   Scale factor for y.
 * @param[in,out] y  Device pointer to destination vector.
 * @param[in] incy   Stride for y.
 */
template <typename T>
EINSUMS_EXPORT void axpby(int64_t n, T alpha, T const *x, int64_t incx, T beta, T *y, int64_t incy);

/**
 * @brief Dot product on device memory: result = x . y.
 * @param[in] n    Number of elements.
 * @param[in] x    Device pointer to first vector.
 * @param[in] incx Stride for x.
 * @param[in] y    Device pointer to second vector.
 * @param[in] incy Stride for y.
 * @return The dot product.
 */
template <typename T>
EINSUMS_EXPORT T dot(int64_t n, T const *x, int64_t incx, T const *y, int64_t incy);

/**
 * @brief L2 norm on device memory: result = ||x||_2.
 * @param[in] n    Number of elements.
 * @param[in] x    Device pointer to the vector.
 * @param[in] incx Stride for x.
 * @return The Euclidean norm.
 */
template <typename T>
EINSUMS_EXPORT T nrm2(int64_t n, T const *x, int64_t incx);

// ===========================================================================
// Reduced-precision GEMM for Ozaki optimization pass.
// FP16/FP8 inputs with FP32 accumulation via tensor cores.
// ===========================================================================

/// FP16 GEMM: C(fp32) = alpha * A(fp16) * B(fp16) + beta * C(fp32)
EINSUMS_EXPORT void hgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, half_t const *a, int64_t lda,
                          half_t const *b, int64_t ldb, float beta, float *c, int64_t ldc);

/// BFloat16 GEMM: C(fp32) = alpha * A(bf16) * B(bf16) + beta * C(fp32)
EINSUMS_EXPORT void bfgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, bfloat16_t const *a, int64_t lda,
                           bfloat16_t const *b, int64_t ldb, float beta, float *c, int64_t ldc);

/// FP8 GEMM: C(fp32) = alpha * A(fp8) * B(fp8) + beta * C(fp32)
EINSUMS_EXPORT void fp8gemm(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, fp8_t const *a, int64_t lda,
                            fp8_t const *b, int64_t ldb, float beta, float *c, int64_t ldc);

} // namespace einsums::gpu::blas
