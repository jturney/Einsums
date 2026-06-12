//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/GPU/Platform.hpp>

#if defined(EINSUMS_HAVE_MPS)

#    include <cstddef>
#    include <string>

namespace einsums::gpu::mps {

// Memory management
EINSUMS_EXPORT void  *device_malloc(size_t bytes);
EINSUMS_EXPORT void   device_free(void *ptr);
EINSUMS_EXPORT void   memcpy_host_to_device(void *dst, void const *src, size_t bytes);
EINSUMS_EXPORT void   memcpy_device_to_host(void *dst, void const *src, size_t bytes);
EINSUMS_EXPORT void   memcpy_device_to_device(void *dst, void const *src, size_t bytes);
EINSUMS_EXPORT void   device_memset(void *ptr, int value, size_t bytes);
EINSUMS_EXPORT void   device_synchronize();
EINSUMS_EXPORT size_t available_device_memory();
EINSUMS_EXPORT std::string device_name();

// BLAS: GEMM
EINSUMS_EXPORT void sgemm(char transa, char transb, int m, int n, int k, float alpha, float const *a, int lda, float const *b, int ldb,
                          float beta, float *c, int ldc);

/// FP16 GEMM: all matrices are __fp16, alpha/beta are float.
EINSUMS_EXPORT void hgemm(char transa, char transb, int m, int n, int k, float alpha, __fp16 const *a, int lda, __fp16 const *b, int ldb,
                          float beta, __fp16 *c, int ldc);

// Note: MPSMatrixMultiplication does not support BFloat16 or ComplexFloat32/ComplexFloat16.
// BFloat16 GEMM converts BF16→F32, runs MPS sgemm, stores F32 result.
// Complex types fall back to CPU BLAS.

// BLAS: GEMV
EINSUMS_EXPORT void sgemv(char trans, int m, int n, float alpha, float const *a, int lda, float const *x, int incx, float beta, float *y,
                          int incy);

} // namespace einsums::gpu::mps

#endif // EINSUMS_HAVE_MPS
