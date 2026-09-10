//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/GPU/BLAS.hpp>
#include <Einsums/GPU/Error.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/GPU/Stream.hpp>
#include <Einsums/GPU/Types.hpp>

#include <vector>

#if defined(EINSUMS_HAVE_CUDA)
#    include <cublas_v2.h>
#elif defined(EINSUMS_HAVE_HIP)
#    include <hipblas/hipblas.h>
#elif defined(EINSUMS_HAVE_MPS)
#    include <Einsums/GPU/MPSBackend.hpp>
// MPS only supports float32 GEMM; other types fall back to CPU BLAS
#    include <Einsums/BLASVendor/Vendor.hpp>
#else
// Mock backend: delegate to CPU BLAS
#    include <Einsums/BLASVendor/Vendor.hpp>
#endif

// The strided-batched CPU fallback needs BLASVendor unconditionally
// (the CUDA/HIP branches use cuBLAS/hipBLAS directly, but we still
// ship the non-GPU fallback body so mock builds compile cleanly).
#if !defined(EINSUMS_HAVE_CUDA) && !defined(EINSUMS_HAVE_HIP) && !defined(EINSUMS_HAVE_MPS)
// Already covered by the mock include above.
#elif defined(EINSUMS_HAVE_MPS)
// Already covered by the MPS include above.
#endif

EINSUMS_NAMESPACE_BEGIN(gpu::blas)

namespace {
/// Unwrap device_malloc or throw, for internal BLAS use where OOM is fatal.
void *device_malloc_or_throw(size_t bytes) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    auto result = gpu::device_malloc(bytes);
    if (!result)
        throw std::runtime_error(result.error().message);
    return result.value();
}
} // namespace

// ===========================================================================
// Standard precision GEMM
// ===========================================================================

void sgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float const *a, int64_t lda, float const *b, int64_t ldb,
           float beta, float *c, int64_t ldc) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(cublasSgemm_v2(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k, &alpha, a,
                                  lda, b, ldb, &beta, c, ldc));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasSgemm(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k, &alpha, a,
                                lda, b, ldb, &beta, c, ldc));
#elif defined(EINSUMS_HAVE_MPS)
    mps::sgemm(transa, transb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha, a, static_cast<int>(lda), b,
               static_cast<int>(ldb), beta, c, static_cast<int>(ldc));
#else
    ::einsums::blas::vendor::sgemm(transa, transb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha, a,
                                   static_cast<int>(lda), b, static_cast<int>(ldb), beta, c, static_cast<int>(ldc));
#endif
}

void dgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double const *a, int64_t lda, double const *b,
           int64_t ldb, double beta, double *c, int64_t ldc) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(cublasDgemm_v2(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k, &alpha, a,
                                  lda, b, ldb, &beta, c, ldc));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasDgemm(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k, &alpha, a,
                                lda, b, ldb, &beta, c, ldc));
#else
    // MPS and mock: no GPU double precision, fall back to CPU BLAS.
    ::einsums::blas::vendor::dgemm(transa, transb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha, a,
                                   static_cast<int>(lda), b, static_cast<int>(ldb), beta, c, static_cast<int>(ldc));
#endif
}

void cgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, std::complex<float> alpha, std::complex<float> const *a, int64_t lda,
           std::complex<float> const *b, int64_t ldb, std::complex<float> beta, std::complex<float> *c, int64_t ldc) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(cublasCgemm_v2(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k,
                                  reinterpret_cast<cuComplex const *>(&alpha), reinterpret_cast<cuComplex const *>(a), lda,
                                  reinterpret_cast<cuComplex const *>(b), ldb, reinterpret_cast<cuComplex const *>(&beta),
                                  reinterpret_cast<cuComplex *>(c), ldc));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasCgemm(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k,
                                reinterpret_cast<hipblasComplex const *>(&alpha), reinterpret_cast<hipblasComplex const *>(a), lda,
                                reinterpret_cast<hipblasComplex const *>(b), ldb, reinterpret_cast<hipblasComplex const *>(&beta),
                                reinterpret_cast<hipblasComplex *>(c), ldc));
#elif defined(EINSUMS_HAVE_MPS)
    // MPS ComplexFloat32 GEMM is not reliably supported by MPSMatrixMultiplication.
    // Fall back to CPU BLAS for complex types.
    ::einsums::blas::vendor::cgemm(transa, transb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha, a,
                                   static_cast<int>(lda), b, static_cast<int>(ldb), beta, c, static_cast<int>(ldc));
#else
    ::einsums::blas::vendor::cgemm(transa, transb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha, a,
                                   static_cast<int>(lda), b, static_cast<int>(ldb), beta, c, static_cast<int>(ldc));
#endif
}

void zgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, std::complex<double> alpha, std::complex<double> const *a,
           int64_t lda, std::complex<double> const *b, int64_t ldb, std::complex<double> beta, std::complex<double> *c, int64_t ldc) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(cublasZgemm_v2(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k,
                                  reinterpret_cast<cuDoubleComplex const *>(&alpha), reinterpret_cast<cuDoubleComplex const *>(a), lda,
                                  reinterpret_cast<cuDoubleComplex const *>(b), ldb, reinterpret_cast<cuDoubleComplex const *>(&beta),
                                  reinterpret_cast<cuDoubleComplex *>(c), ldc));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasZgemm(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k,
                                reinterpret_cast<hipblasDoubleComplex const *>(&alpha), reinterpret_cast<hipblasDoubleComplex const *>(a),
                                lda, reinterpret_cast<hipblasDoubleComplex const *>(b), ldb,
                                reinterpret_cast<hipblasDoubleComplex const *>(&beta), reinterpret_cast<hipblasDoubleComplex *>(c), ldc));
#else
    // MPS and mock: CPU fallback.
    ::einsums::blas::vendor::zgemm(transa, transb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha, a,
                                   static_cast<int>(lda), b, static_cast<int>(ldb), beta, c, static_cast<int>(ldc));
#endif
}

// ===========================================================================
// Strided-batched GEMM.
//
// On CUDA/HIP: one `{cublas,hipblas}?gemmStridedBatched` call handles
// the entire batch; the GPU runtime pipelines matrix launches and
// avoids per-call pointer-array construction.
//
// On CPU / mock / MPS: fall back to the pointer-array CPU
// `blas::vendor::?gemm_batch` by computing pointer arrays from the
// base + stride info. Result is identical; performance matches the
// non-batched path because there's no GPU to exploit.
//
// TODO: Apple MPS has native batched matmul via
// `MPSMatrixMultiplication` (`batchStart` / `batchSize`) and
// `MPSNDArrayMatrixMultiplication` (macOS 13+). Replacing the MPS
// fall-through with a real MPS batched dispatch would give Apple
// Silicon users GPU-accelerated batching without them having to
// rebuild against CUDA/HIP. Blockers: MPS expects row-major storage
// so the col-major-batch-last layout needs a thin wrapper, and MPS
// doubles generally round-trip to CPU anyway, so float32 is the
// first target.
//
// Pattern shared across all dtypes, kept inline rather than factored
// into a helper because each type has its own cuBLAS entry point.
// ===========================================================================

namespace {

// Build pointer arrays (host-side) from base + stride for the CPU fallback.
template <typename T>
void make_ptr_arrays(T const *base_a, T const *base_b, T *base_c, int64_t stride_a, int64_t stride_b, int64_t stride_c, int64_t batch_count,
                     std::vector<T const *> &a_arr, std::vector<T const *> &b_arr, std::vector<T *> &c_arr) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    a_arr.resize(static_cast<size_t>(batch_count));
    b_arr.resize(static_cast<size_t>(batch_count));
    c_arr.resize(static_cast<size_t>(batch_count));
    for (int64_t i = 0; i < batch_count; ++i) {
        a_arr[static_cast<size_t>(i)] = base_a + i * stride_a;
        b_arr[static_cast<size_t>(i)] = base_b + i * stride_b;
        c_arr[static_cast<size_t>(i)] = base_c + i * stride_c;
    }
}

} // namespace

void sgemm_strided_batched(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float const *a, int64_t lda,
                           int64_t stride_a, float const *b, int64_t ldb, int64_t stride_b, float beta, float *c, int64_t ldc,
                           int64_t stride_c, int64_t batch_count) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(cublasSgemmStridedBatched(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k,
                                             &alpha, a, lda, stride_a, b, ldb, stride_b, &beta, c, ldc, stride_c, batch_count));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasSgemmStridedBatched(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n,
                                              k, &alpha, a, lda, stride_a, b, ldb, stride_b, &beta, c, ldc, stride_c, batch_count));
#else
    std::vector<float const *> a_arr, b_arr;
    std::vector<float *>       c_arr;
    make_ptr_arrays(a, b, c, stride_a, stride_b, stride_c, batch_count, a_arr, b_arr, c_arr);
    ::einsums::blas::vendor::sgemm_batch(transa, transb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha, a_arr.data(),
                                         static_cast<int>(lda), b_arr.data(), static_cast<int>(ldb), beta, c_arr.data(),
                                         static_cast<int>(ldc), static_cast<int>(batch_count));
#endif
}

void dgemm_strided_batched(char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double const *a, int64_t lda,
                           int64_t stride_a, double const *b, int64_t ldb, int64_t stride_b, double beta, double *c, int64_t ldc,
                           int64_t stride_c, int64_t batch_count) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(cublasDgemmStridedBatched(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k,
                                             &alpha, a, lda, stride_a, b, ldb, stride_b, &beta, c, ldc, stride_c, batch_count));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasDgemmStridedBatched(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n,
                                              k, &alpha, a, lda, stride_a, b, ldb, stride_b, &beta, c, ldc, stride_c, batch_count));
#else
    // MPS has no strided-batched dgemm; fall through to CPU pointer-array batch.
    std::vector<double const *> a_arr, b_arr;
    std::vector<double *>       c_arr;
    make_ptr_arrays(a, b, c, stride_a, stride_b, stride_c, batch_count, a_arr, b_arr, c_arr);
    ::einsums::blas::vendor::dgemm_batch(transa, transb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha, a_arr.data(),
                                         static_cast<int>(lda), b_arr.data(), static_cast<int>(ldb), beta, c_arr.data(),
                                         static_cast<int>(ldc), static_cast<int>(batch_count));
#endif
}

void cgemm_strided_batched(char transa, char transb, int64_t m, int64_t n, int64_t k, std::complex<float> alpha,
                           std::complex<float> const *a, int64_t lda, int64_t stride_a, std::complex<float> const *b, int64_t ldb,
                           int64_t stride_b, std::complex<float> beta, std::complex<float> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_count) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(cublasCgemmStridedBatched(get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k,
                                             reinterpret_cast<cuComplex const *>(&alpha), reinterpret_cast<cuComplex const *>(a), lda,
                                             stride_a, reinterpret_cast<cuComplex const *>(b), ldb, stride_b,
                                             reinterpret_cast<cuComplex const *>(&beta), reinterpret_cast<cuComplex *>(c), ldc, stride_c,
                                             batch_count));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasCgemmStridedBatched(
        get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k,
        reinterpret_cast<hipblasComplex const *>(&alpha), reinterpret_cast<hipblasComplex const *>(a), lda, stride_a,
        reinterpret_cast<hipblasComplex const *>(b), ldb, stride_b, reinterpret_cast<hipblasComplex const *>(&beta),
        reinterpret_cast<hipblasComplex *>(c), ldc, stride_c, batch_count));
#else
    std::vector<std::complex<float> const *> a_arr, b_arr;
    std::vector<std::complex<float> *>       c_arr;
    make_ptr_arrays(a, b, c, stride_a, stride_b, stride_c, batch_count, a_arr, b_arr, c_arr);
    ::einsums::blas::vendor::cgemm_batch(transa, transb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha, a_arr.data(),
                                         static_cast<int>(lda), b_arr.data(), static_cast<int>(ldb), beta, c_arr.data(),
                                         static_cast<int>(ldc), static_cast<int>(batch_count));
#endif
}

void zgemm_strided_batched(char transa, char transb, int64_t m, int64_t n, int64_t k, std::complex<double> alpha,
                           std::complex<double> const *a, int64_t lda, int64_t stride_a, std::complex<double> const *b, int64_t ldb,
                           int64_t stride_b, std::complex<double> beta, std::complex<double> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_count) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(cublasZgemmStridedBatched(
        get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k,
        reinterpret_cast<cuDoubleComplex const *>(&alpha), reinterpret_cast<cuDoubleComplex const *>(a), lda, stride_a,
        reinterpret_cast<cuDoubleComplex const *>(b), ldb, stride_b, reinterpret_cast<cuDoubleComplex const *>(&beta),
        reinterpret_cast<cuDoubleComplex *>(c), ldc, stride_c, batch_count));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasZgemmStridedBatched(
        get_blas_handle(), to_vendor_op(char_to_op(transa)), to_vendor_op(char_to_op(transb)), m, n, k,
        reinterpret_cast<hipblasDoubleComplex const *>(&alpha), reinterpret_cast<hipblasDoubleComplex const *>(a), lda, stride_a,
        reinterpret_cast<hipblasDoubleComplex const *>(b), ldb, stride_b, reinterpret_cast<hipblasDoubleComplex const *>(&beta),
        reinterpret_cast<hipblasDoubleComplex *>(c), ldc, stride_c, batch_count));
#else
    std::vector<std::complex<double> const *> a_arr, b_arr;
    std::vector<std::complex<double> *>       c_arr;
    make_ptr_arrays(a, b, c, stride_a, stride_b, stride_c, batch_count, a_arr, b_arr, c_arr);
    ::einsums::blas::vendor::zgemm_batch(transa, transb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha, a_arr.data(),
                                         static_cast<int>(lda), b_arr.data(), static_cast<int>(ldb), beta, c_arr.data(),
                                         static_cast<int>(ldc), static_cast<int>(batch_count));
#endif
}

// ===========================================================================
// Standard precision GEMV
// ===========================================================================

void sgemv(char trans, int64_t m, int64_t n, float alpha, float const *a, int64_t lda, float const *x, int64_t incx, float beta, float *y,
           int64_t incy) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(cublasSgemv_v2(get_blas_handle(), to_vendor_op(char_to_op(trans)), m, n, &alpha, a, lda, x, incx, &beta, y, incy));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasSgemv(get_blas_handle(), to_vendor_op(char_to_op(trans)), m, n, &alpha, a, lda, x, incx, &beta, y, incy));
#elif defined(EINSUMS_HAVE_MPS)
    mps::sgemv(trans, static_cast<int>(m), static_cast<int>(n), alpha, a, static_cast<int>(lda), x, static_cast<int>(incx), beta, y,
               static_cast<int>(incy));
#else
    ::einsums::blas::vendor::sgemv(trans, static_cast<int>(m), static_cast<int>(n), alpha, a, static_cast<int>(lda), x,
                                   static_cast<int>(incx), beta, y, static_cast<int>(incy));
#endif
}

void dgemv(char trans, int64_t m, int64_t n, double alpha, double const *a, int64_t lda, double const *x, int64_t incx, double beta,
           double *y, int64_t incy) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(cublasDgemv_v2(get_blas_handle(), to_vendor_op(char_to_op(trans)), m, n, &alpha, a, lda, x, incx, &beta, y, incy));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasDgemv(get_blas_handle(), to_vendor_op(char_to_op(trans)), m, n, &alpha, a, lda, x, incx, &beta, y, incy));
#else
    ::einsums::blas::vendor::dgemv(trans, static_cast<int>(m), static_cast<int>(n), alpha, a, static_cast<int>(lda), x,
                                   static_cast<int>(incx), beta, y, static_cast<int>(incy));
#endif
}

template <>
EINSUMS_EXPORT void gemv<float>(char trans, int64_t m, int64_t n, float alpha, float const *a, int64_t lda, float const *x, int64_t incx,
                                float beta, float *y, int64_t incy) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
EINSUMS_EXPORT void gemv<double>(char trans, int64_t m, int64_t n, double alpha, double const *a, int64_t lda, double const *x,
                                 int64_t incx, double beta, double *y, int64_t incy) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

// ===========================================================================
// BLAS Level 1: element-wise operations on device memory.
//
// These used to call ::einsums::blas::vendor::* unconditionally, on every
// backend. That was wrong in the two ways that matter:
//
//   * On CUDA/HIP the pointers are device pointers (the ComputeGraph executor
//     swaps a tensor's data pointer to its device shadow before dispatch), so a
//     host BLAS call on them is a segfault, not a slow path. The old comment
//     here claimed the data was "accessible on all backends, with unified
//     memory on MPS and shadow memory on CUDA/HIP/mock" - shadow memory on
//     CUDA is cudaMalloc'd device memory and is not host-accessible.
//   * The CUDA/HIP branches of this file do not even include BLASVendor, so
//     the code could not compile once EINSUMS_HAVE_CUDA was defined.
//
// On MPS and mock the host fallback is correct: MPS is unified memory and the
// mock "device" is plain malloc.
// ===========================================================================

template <>
EINSUMS_EXPORT void scal<float>(int64_t n, float alpha, float *x, int64_t incx) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(cublasSscal_v2(get_blas_handle(), static_cast<int>(n), &alpha, x, static_cast<int>(incx)));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasSscal(get_blas_handle(), static_cast<int>(n), &alpha, x, static_cast<int>(incx)));
#else
    ::einsums::blas::vendor::sscal(static_cast<int>(n), alpha, x, static_cast<int>(incx));
#endif
}

template <>
EINSUMS_EXPORT void scal<double>(int64_t n, double alpha, double *x, int64_t incx) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(cublasDscal_v2(get_blas_handle(), static_cast<int>(n), &alpha, x, static_cast<int>(incx)));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasDscal(get_blas_handle(), static_cast<int>(n), &alpha, x, static_cast<int>(incx)));
#else
    ::einsums::blas::vendor::dscal(static_cast<int>(n), alpha, x, static_cast<int>(incx));
#endif
}

template <>
EINSUMS_EXPORT void axpy<float>(int64_t n, float alpha, float const *x, int64_t incx, float *y, int64_t incy) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(
        cublasSaxpy_v2(get_blas_handle(), static_cast<int>(n), &alpha, x, static_cast<int>(incx), y, static_cast<int>(incy)));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasSaxpy(get_blas_handle(), static_cast<int>(n), &alpha, x, static_cast<int>(incx), y, static_cast<int>(incy)));
#else
    ::einsums::blas::vendor::saxpy(static_cast<int>(n), alpha, x, static_cast<int>(incx), y, static_cast<int>(incy));
#endif
}

template <>
EINSUMS_EXPORT void axpy<double>(int64_t n, double alpha, double const *x, int64_t incx, double *y, int64_t incy) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_blas_catch(
        cublasDaxpy_v2(get_blas_handle(), static_cast<int>(n), &alpha, x, static_cast<int>(incx), y, static_cast<int>(incy)));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_blas_catch(hipblasDaxpy(get_blas_handle(), static_cast<int>(n), &alpha, x, static_cast<int>(incx), y, static_cast<int>(incy)));
#else
    ::einsums::blas::vendor::daxpy(static_cast<int>(n), alpha, x, static_cast<int>(incx), y, static_cast<int>(incy));
#endif
}

// axpby has no single cuBLAS/hipBLAS entry point: scale y by beta, then axpy.
// Both calls are enqueued on the same stream, so the ordering is guaranteed.
template <>
EINSUMS_EXPORT void axpby<float>(int64_t n, float alpha, float const *x, int64_t incx, float beta, float *y, int64_t incy) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    scal<float>(n, beta, y, incy);
    axpy<float>(n, alpha, x, incx, y, incy);
}

template <>
EINSUMS_EXPORT void axpby<double>(int64_t n, double alpha, double const *x, int64_t incx, double beta, double *y, int64_t incy) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    scal<double>(n, beta, y, incy);
    axpy<double>(n, alpha, x, incx, y, incy);
}

// dot and nrm2 return a scalar to the host. cuBLAS/hipBLAS default to
// CUBLAS_POINTER_MODE_HOST, under which these calls write through a host
// pointer and synchronize before returning, so no explicit sync is needed.
template <>
EINSUMS_EXPORT float dot<float>(int64_t n, float const *x, int64_t incx, float const *y, int64_t incy) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    float result = 0.0F;
    gpu_blas_catch(
        cublasSdot_v2(get_blas_handle(), static_cast<int>(n), x, static_cast<int>(incx), y, static_cast<int>(incy), &result));
    return result;
#elif defined(EINSUMS_HAVE_HIP)
    float result = 0.0F;
    gpu_blas_catch(hipblasSdot(get_blas_handle(), static_cast<int>(n), x, static_cast<int>(incx), y, static_cast<int>(incy), &result));
    return result;
#else
    return ::einsums::blas::vendor::sdot(static_cast<int>(n), x, static_cast<int>(incx), y, static_cast<int>(incy));
#endif
}

template <>
EINSUMS_EXPORT double dot<double>(int64_t n, double const *x, int64_t incx, double const *y, int64_t incy) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    double result = 0.0;
    gpu_blas_catch(
        cublasDdot_v2(get_blas_handle(), static_cast<int>(n), x, static_cast<int>(incx), y, static_cast<int>(incy), &result));
    return result;
#elif defined(EINSUMS_HAVE_HIP)
    double result = 0.0;
    gpu_blas_catch(hipblasDdot(get_blas_handle(), static_cast<int>(n), x, static_cast<int>(incx), y, static_cast<int>(incy), &result));
    return result;
#else
    return ::einsums::blas::vendor::ddot(static_cast<int>(n), x, static_cast<int>(incx), y, static_cast<int>(incy));
#endif
}

template <>
EINSUMS_EXPORT float nrm2<float>(int64_t n, float const *x, int64_t incx) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    float result = 0.0F;
    gpu_blas_catch(cublasSnrm2_v2(get_blas_handle(), static_cast<int>(n), x, static_cast<int>(incx), &result));
    return result;
#elif defined(EINSUMS_HAVE_HIP)
    float result = 0.0F;
    gpu_blas_catch(hipblasSnrm2(get_blas_handle(), static_cast<int>(n), x, static_cast<int>(incx), &result));
    return result;
#else
    return ::einsums::blas::vendor::snrm2(static_cast<int>(n), x, static_cast<int>(incx));
#endif
}

template <>
EINSUMS_EXPORT double nrm2<double>(int64_t n, double const *x, int64_t incx) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    double result = 0.0;
    gpu_blas_catch(cublasDnrm2_v2(get_blas_handle(), static_cast<int>(n), x, static_cast<int>(incx), &result));
    return result;
#elif defined(EINSUMS_HAVE_HIP)
    double result = 0.0;
    gpu_blas_catch(hipblasDnrm2(get_blas_handle(), static_cast<int>(n), x, static_cast<int>(incx), &result));
    return result;
#else
    return ::einsums::blas::vendor::dnrm2(static_cast<int>(n), x, static_cast<int>(incx));
#endif
}

// ===========================================================================
// Template wrapper: explicit instantiations (GEMM)
// ===========================================================================

template <>
EINSUMS_EXPORT void gemm<float>(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float const *a, int64_t lda,
                                float const *b, int64_t ldb, float beta, float *c, int64_t ldc) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
EINSUMS_EXPORT void gemm<double>(char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double const *a, int64_t lda,
                                 double const *b, int64_t ldb, double beta, double *c, int64_t ldc) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
EINSUMS_EXPORT void gemm<std::complex<float>>(char transa, char transb, int64_t m, int64_t n, int64_t k, std::complex<float> alpha,
                                              std::complex<float> const *a, int64_t lda, std::complex<float> const *b, int64_t ldb,
                                              std::complex<float> beta, std::complex<float> *c, int64_t ldc) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
EINSUMS_EXPORT void gemm<std::complex<double>>(char transa, char transb, int64_t m, int64_t n, int64_t k, std::complex<double> alpha,
                                               std::complex<double> const *a, int64_t lda, std::complex<double> const *b, int64_t ldb,
                                               std::complex<double> beta, std::complex<double> *c, int64_t ldc) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
EINSUMS_EXPORT void gemm_strided_batched<float>(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float const *a,
                                                int64_t lda, int64_t stride_a, float const *b, int64_t ldb, int64_t stride_b, float beta,
                                                float *c, int64_t ldc, int64_t stride_c, int64_t batch_count) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    sgemm_strided_batched(transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count);
}

template <>
EINSUMS_EXPORT void gemm_strided_batched<double>(char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double const *a,
                                                 int64_t lda, int64_t stride_a, double const *b, int64_t ldb, int64_t stride_b, double beta,
                                                 double *c, int64_t ldc, int64_t stride_c, int64_t batch_count) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    dgemm_strided_batched(transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count);
}

template <>
EINSUMS_EXPORT void gemm_strided_batched<std::complex<float>>(char transa, char transb, int64_t m, int64_t n, int64_t k,
                                                              std::complex<float> alpha, std::complex<float> const *a, int64_t lda,
                                                              int64_t stride_a, std::complex<float> const *b, int64_t ldb, int64_t stride_b,
                                                              std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                                                              int64_t stride_c, int64_t batch_count) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    cgemm_strided_batched(transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count);
}

template <>
EINSUMS_EXPORT void gemm_strided_batched<std::complex<double>>(char transa, char transb, int64_t m, int64_t n, int64_t k,
                                                               std::complex<double> alpha, std::complex<double> const *a, int64_t lda,
                                                               int64_t stride_a, std::complex<double> const *b, int64_t ldb,
                                                               int64_t stride_b, std::complex<double> beta, std::complex<double> *c,
                                                               int64_t ldc, int64_t stride_c, int64_t batch_count) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
    zgemm_strided_batched(transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count);
}

// ===========================================================================
// Reduced-precision GEMM: stubs for now, implemented when Ozaki pass lands
// ===========================================================================

void hgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, half_t const *a, int64_t lda, half_t const *b,
           int64_t ldb, float beta, float *c, [[maybe_unused]] int64_t ldc) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    // TODO: cublasLtMatmul. FP16 wants CUDA_R_16F in / CUBLAS_COMPUTE_32F; FP8 E4M3
    // additionally needs sm_89+. Throws until then rather than leaving C untouched.
    not_implemented("gpu::blas::hgemm");
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::blas::hgemm");
#elif defined(EINSUMS_HAVE_MPS)
    // MPS pure FP16 GEMM: C_fp16 = alpha * A_fp16 * B_fp16 + beta * C_fp16.
    // Our signature has float* C output. We compute in FP16 into a temp buffer, then convert.
    {
        int64_t c_size = m * n;
        auto   *c_fp16 = static_cast<half_t *>(device_malloc_or_throw(static_cast<size_t>(c_size) * sizeof(half_t)));

        // Convert existing C (float) to FP16 for the beta accumulation.
        if (beta != 0.0f) {
            for (int64_t i = 0; i < c_size; i++) {
                c_fp16[i] = static_cast<half_t>(c[i]);
            }
        } else {
            gpu::device_memset(c_fp16, 0, static_cast<size_t>(c_size) * sizeof(half_t));
        }

        mps::hgemm(transa, transb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha, a, static_cast<int>(lda), b,
                   static_cast<int>(ldb), beta, c_fp16, static_cast<int>(m));

        // Convert FP16 result back to float.
        for (int64_t i = 0; i < c_size; i++) {
            c[i] = static_cast<float>(c_fp16[i]);
        }

        gpu::device_free(c_fp16);
    }
#else
    // Mock: no FP16 hardware
    (void)transa;
    (void)transb;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)beta;
    (void)c;
    (void)ldc;
#endif
}

void bfgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, bfloat16_t const *a, int64_t lda, bfloat16_t const *b,
            int64_t ldb, float beta, float *c, int64_t ldc) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_MPS)
    // MPSMatrixMultiplication does NOT support BFloat16 (crashes with SIGABRT).
    // Fallback: convert BF16 inputs to Float32, compute sgemm, then store to Float32 output.
    {
        int64_t a_size = lda * ((transa == 'n' || transa == 'N') ? k : m);
        int64_t b_size = ldb * ((transb == 'n' || transb == 'N') ? n : k);

        auto *a_f32 = static_cast<float *>(device_malloc_or_throw(static_cast<size_t>(a_size) * sizeof(float)));
        auto *b_f32 = static_cast<float *>(device_malloc_or_throw(static_cast<size_t>(b_size) * sizeof(float)));

        for (int64_t i = 0; i < a_size; i++)
            a_f32[i] = static_cast<float>(a[i]);
        for (int64_t i = 0; i < b_size; i++)
            b_f32[i] = static_cast<float>(b[i]);

        // Use MPS Float32 GEMM on the converted data.
        mps::sgemm(transa, transb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha, a_f32, static_cast<int>(lda),
                   b_f32, static_cast<int>(ldb), beta, c, static_cast<int>(ldc));

        gpu::device_free(a_f32);
        gpu::device_free(b_f32);
    }
#else
    // No BFloat16 support on CUDA/HIP/mock.
    (void)transa;
    (void)transb;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)beta;
    (void)c;
    (void)ldc;
#endif
}

void fp8gemm(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, fp8_t const *a, int64_t lda, fp8_t const *b,
             int64_t ldb, float beta, float *c, int64_t ldc) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    // TODO: cublasLtMatmul. FP16 wants CUDA_R_16F in / CUBLAS_COMPUTE_32F; FP8 E4M3
    // additionally needs sm_89+. Throws until then rather than leaving C untouched.
    not_implemented("gpu::blas::fp8gemm");
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::blas::fp8gemm");
#else
    // Mock: no FP8 hardware
    (void)transa;
    (void)transb;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)beta;
    (void)c;
    (void)ldc;
#endif
}

EINSUMS_NAMESPACE_END(gpu::blas)
