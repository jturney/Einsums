//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Metal Performance Shaders backend for Apple Silicon GPU.
// This file is compiled as Objective-C++ (.mm) to access the Metal API.

#include <Einsums/Config.hpp>
#include <Einsums/GPU/Platform.hpp>

#if defined(EINSUMS_HAVE_MPS)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstddef>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace einsums::gpu::mps {

// ===========================================================================
// Global Metal state — initialized lazily on first use.
// ===========================================================================

namespace {

id<MTLDevice>       g_device       = nil;
id<MTLCommandQueue> g_command_queue = nil;
std::once_flag      g_init_flag;

/// Map from raw pointer (.contents) → MTLBuffer for lookup during GEMM.
std::mutex                                    g_buffer_mutex;
std::unordered_map<void *, id<MTLBuffer>>     g_buffer_map;

void ensure_initialized() {
    std::call_once(g_init_flag, []() {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            // Should not happen on Apple Silicon
            return;
        }
        g_command_queue = [g_device newCommandQueue];
    });
}

} // namespace

// ===========================================================================
// Device info
// ===========================================================================

id<MTLDevice> get_device() {
    ensure_initialized();
    return g_device;
}

id<MTLCommandQueue> get_command_queue() {
    ensure_initialized();
    return g_command_queue;
}

// ===========================================================================
// Memory management
// ===========================================================================

void *device_malloc(size_t bytes) {
    ensure_initialized();
    if (!g_device || bytes == 0)
        return nullptr;

    id<MTLBuffer> buffer = [g_device newBufferWithLength:bytes
                                                options:MTLResourceStorageModeShared];
    if (!buffer)
        return nullptr;

    void *ptr = buffer.contents;

    {
        std::lock_guard lock(g_buffer_mutex);
        g_buffer_map[ptr] = buffer;
    }

    return ptr;
}

void device_free(void *ptr) {
    if (!ptr)
        return;

    std::lock_guard lock(g_buffer_mutex);
    auto it = g_buffer_map.find(ptr);
    if (it != g_buffer_map.end()) {
        // ARC will release the MTLBuffer when we remove the reference
        g_buffer_map.erase(it);
    }
}

id<MTLBuffer> find_buffer(void const *ptr) {
    std::lock_guard lock(g_buffer_mutex);
    auto it = g_buffer_map.find(const_cast<void *>(ptr));
    if (it != g_buffer_map.end())
        return it->second;
    return nil;
}

void memcpy_host_to_device(void *dst, void const *src, size_t bytes) {
    // On Apple Silicon unified memory, both host and device pointers
    // are CPU-accessible. A simple memcpy moves data between allocations.
    std::memcpy(dst, src, bytes);
}

void memcpy_device_to_host(void *dst, void const *src, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void memcpy_device_to_device(void *dst, void const *src, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void device_memset(void *ptr, int value, size_t bytes) {
    std::memset(ptr, value, bytes);
}

void device_synchronize() {
    // On MPS, synchronization happens per-command-buffer.
    // This is a global sync point — no pending work on mock-style usage.
}

size_t available_device_memory() {
    ensure_initialized();
    if (!g_device)
        return 0;
    // Apple Silicon: report recommended working set size (typically ~2/3 of unified RAM).
    return g_device.recommendedMaxWorkingSetSize;
}

std::string device_name() {
    ensure_initialized();
    if (!g_device)
        return "Unknown MPS Device";
    return std::string([g_device.name UTF8String]);
}

// ===========================================================================
// GEMM via MPSMatrixMultiplication — generic implementation for all types
// ===========================================================================

/// Wrap a pointer in an MTLBuffer. If the pointer comes from device_malloc,
/// we already have a buffer in g_buffer_map. Otherwise, try newBufferWithBytesNoCopy
/// (requires page-aligned pointer). If alignment fails, copy into a new buffer.
id<MTLBuffer> wrap_or_copy(void const *ptr, size_t bytes, bool is_const) {
    id<MTLBuffer> buf = find_buffer(ptr);
    if (buf)
        return buf;

    // Try zero-copy wrapping (requires page-aligned pointer and page-aligned length).
    NSUInteger pageSize = getpagesize();
    bool aligned = (reinterpret_cast<uintptr_t>(ptr) % pageSize) == 0;
    // Round up length to page boundary for newBufferWithBytesNoCopy.
    NSUInteger alignedLen = ((bytes + pageSize - 1) / pageSize) * pageSize;

    if (aligned) {
        buf = [g_device newBufferWithBytesNoCopy:const_cast<void *>(ptr)
                                          length:alignedLen
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
        if (buf)
            return buf;
    }

    // Fallback: allocate a new buffer and copy data into it.
    buf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buf) {
        std::memcpy(buf.contents, ptr, bytes);
    }
    return buf;
}

/// Execute a command buffer and check for errors.
bool execute_and_check(id<MTLCommandBuffer> cmdBuf, char const *operation) {
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (cmdBuf.status == MTLCommandBufferStatusError) {
        NSError *err = cmdBuf.error;
        fprintf(stderr, "MPS %s failed: %s (code %ld)\n", operation,
                err ? [err.localizedDescription UTF8String] : "unknown error",
                err ? (long)err.code : -1);
        return false;
    }
    return true;
}

/// Generic MPS GEMM: works for Float32, Float16, BFloat16.
bool mps_gemm_impl(char transa, char transb, int m, int n, int k,
                    double alpha, void const *a, int lda,
                    void const *b, int ldb,
                    double beta, void *c, int ldc,
                    MPSDataType dataType, size_t elementSize) {
    ensure_initialized();
    if (!g_device || !g_command_queue)
        return false;

    size_t sizeA = (size_t)lda * ((transa == 'n' || transa == 'N') ? k : m) * elementSize;
    size_t sizeB = (size_t)ldb * ((transb == 'n' || transb == 'N') ? n : k) * elementSize;
    size_t sizeC = (size_t)ldc * n * elementSize;

    id<MTLBuffer> bufA = wrap_or_copy(a, sizeA, true);
    id<MTLBuffer> bufB = wrap_or_copy(b, sizeB, true);
    id<MTLBuffer> bufC = wrap_or_copy(c, sizeC, false);

    if (!bufA || !bufB || !bufC) {
        fprintf(stderr, "MPS GEMM: failed to create MTLBuffers\n");
        return false;
    }

    // MPS uses row-major convention. BLAS uses column-major.
    // Column-major C = A * B is equivalent to row-major C^T = B^T * A^T.
    // So we swap A↔B and flip the transposes.
    BOOL mpsTransA = (transb == 't' || transb == 'T' || transb == 'c' || transb == 'C') ? YES : NO;
    BOOL mpsTransB = (transa == 't' || transa == 'T' || transa == 'c' || transa == 'C') ? YES : NO;

    NSUInteger rowsB = mpsTransA ? (NSUInteger)k : (NSUInteger)n;
    NSUInteger colsB = mpsTransA ? (NSUInteger)n : (NSUInteger)k;
    NSUInteger rbB   = (NSUInteger)ldb * elementSize;

    NSUInteger rowsA = mpsTransB ? (NSUInteger)m : (NSUInteger)k;
    NSUInteger colsA = mpsTransB ? (NSUInteger)k : (NSUInteger)m;
    NSUInteger rbA   = (NSUInteger)lda * elementSize;

    NSUInteger rowsC = (NSUInteger)n;
    NSUInteger colsC = (NSUInteger)m;
    NSUInteger rbC   = (NSUInteger)ldc * elementSize;

    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:rowsB columns:colsB rowBytes:rbB dataType:dataType];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:rowsA columns:colsA rowBytes:rbA dataType:dataType];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:rowsC columns:colsC rowBytes:rbC dataType:dataType];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descA]; // swapped
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descB]; // swapped
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication *gemm =
        [[MPSMatrixMultiplication alloc] initWithDevice:g_device
                                         transposeLeft:mpsTransA
                                        transposeRight:mpsTransB
                                            resultRows:rowsC
                                         resultColumns:colsC
                                       interiorColumns:k
                                                 alpha:alpha
                                                  beta:beta];

    id<MTLCommandBuffer> cmdBuf = [g_command_queue commandBuffer];
    [gemm encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];

    bool ok = execute_and_check(cmdBuf, "GEMM");

    // If C was copied into a temp buffer (not from g_buffer_map), copy results back.
    if (ok && bufC.contents != c) {
        std::memcpy(c, bufC.contents, sizeC);
    }

    return ok;
}

// --- Type-specific wrappers ---

void sgemm(char transa, char transb, int m, int n, int k,
            float alpha, float const *a, int lda,
            float const *b, int ldb,
            float beta, float *c, int ldc) {
    mps_gemm_impl(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                   MPSDataTypeFloat32, sizeof(float));
}

void hgemm(char transa, char transb, int m, int n, int k,
            float alpha, __fp16 const *a, int lda,
            __fp16 const *b, int ldb,
            float beta, __fp16 *c, int ldc) {
    // Float16 GEMM: inputs and outputs are FP16, alpha/beta are float (promoted to double internally).
    mps_gemm_impl(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                   MPSDataTypeFloat16, sizeof(__fp16));
}

// Note: MPSMatrixMultiplication does NOT support BFloat16 or ComplexFloat32/ComplexFloat16.
// These types exist in MPSDataType but cause SIGABRT when used with matrix multiply.
// Complex GEMM falls back to CPU BLAS in BLAS.cpp.

// ===========================================================================
// GEMV via MPSMatrixVectorMultiplication
// ===========================================================================

void sgemv(char trans, int m, int n,
            float alpha, float const *a, int lda,
            float const *x, int incx,
            float beta, float *y, int incy) {
    ensure_initialized();
    if (!g_device || !g_command_queue)
        return;

    // MPS only supports incx=1 and incy=1. Fall back if strides != 1.
    if (incx != 1 || incy != 1)
        return;

    // BLAS GEMV: y = alpha * op(A) * x + beta * y
    // A is m×n (column-major), x has length n (or m if transposed), y has length m (or n if transposed).
    BOOL transpose = (trans == 't' || trans == 'T' || trans == 'c' || trans == 'C') ? YES : NO;

    // op(A) dimensions: rows × cols
    // No transpose: rows=m, cols=n, x has n elements, y has m elements.
    // Transpose:    rows=n, cols=m, x has m elements, y has n elements.
    NSUInteger rows = transpose ? (NSUInteger)n : (NSUInteger)m;
    NSUInteger cols = transpose ? (NSUInteger)m : (NSUInteger)n;

    // Find or create MTLBuffers.
    size_t sizeA = (size_t)lda * n * sizeof(float);
    size_t sizeX = (transpose ? m : n) * sizeof(float);
    size_t sizeY = (transpose ? n : m) * sizeof(float);

    id<MTLBuffer> bufA = wrap_or_copy(a, sizeA, true);
    id<MTLBuffer> bufX = wrap_or_copy(x, sizeX, true);
    id<MTLBuffer> bufY = wrap_or_copy(y, sizeY, false);

    if (!bufA || !bufX || !bufY)
        return;

    // MPS uses row-major. Column-major A(m×n) with lda=m is row-major A^T(n×m) with rowBytes=lda*sizeof(float).
    // For GEMV without transpose: MPS sees A^T, so we tell MPS transpose=YES to undo it.
    // For GEMV with transpose: MPS sees A^T, and we want A^T, so transpose=NO.
    BOOL mpsTranspose = transpose ? NO : YES;
    NSUInteger mpsRows = (NSUInteger)n;    // rows of the stored row-major matrix (= cols of column-major)
    NSUInteger mpsCols = (NSUInteger)m;    // cols of the stored row-major matrix (= rows of column-major)

    // Matrix descriptor for A in row-major view.
    NSUInteger rbA = (NSUInteger)lda * sizeof(float);
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:mpsRows
                                                                       columns:mpsCols
                                                                      rowBytes:rbA
                                                                      dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];

    // Vector descriptors.
    NSUInteger xLen = transpose ? (NSUInteger)m : (NSUInteger)n;
    NSUInteger yLen = transpose ? (NSUInteger)n : (NSUInteger)m;

    MPSVectorDescriptor *descX = [MPSVectorDescriptor vectorDescriptorWithLength:xLen dataType:MPSDataTypeFloat32];
    MPSVectorDescriptor *descY = [MPSVectorDescriptor vectorDescriptorWithLength:yLen dataType:MPSDataTypeFloat32];

    MPSVector *vecX = [[MPSVector alloc] initWithBuffer:bufX descriptor:descX];
    MPSVector *vecY = [[MPSVector alloc] initWithBuffer:bufY descriptor:descY];

    // MPS GEMV: y = alpha * op(A_mps) * x + beta * y
    // op(A_mps) has dimensions yLen × xLen after transpose.
    MPSMatrixVectorMultiplication *gemv =
        [[MPSMatrixVectorMultiplication alloc] initWithDevice:g_device
                                                   transpose:mpsTranspose
                                                        rows:yLen
                                                     columns:xLen
                                                       alpha:(double)alpha
                                                        beta:(double)beta];

    id<MTLCommandBuffer> cmdBuf = [g_command_queue commandBuffer];
    [gemv encodeToCommandBuffer:cmdBuf inputMatrix:matA inputVector:vecX resultVector:vecY];

    bool ok = execute_and_check(cmdBuf, "GEMV");

    // If Y was copied into a temp buffer, copy results back.
    if (ok && bufY.contents != y) {
        std::memcpy(y, bufY.contents, sizeY);
    }
}

// ===========================================================================
// Stream management
// ===========================================================================

id<MTLCommandQueue> create_command_queue() {
    ensure_initialized();
    return [g_device newCommandQueue];
}

} // namespace einsums::gpu::mps

#endif // EINSUMS_HAVE_MPS
