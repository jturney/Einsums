//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <cstring>
#include <string>

EINSUMS_NAMESPACE_BEGIN(gpu)

/**
 * @brief Structured error for GPU runtime operations.
 *
 * Used with expected<T, GpuError> for recoverable GPU errors
 * (allocation failure, device not found, etc.).
 *
 * BLAS/solver errors continue to throw, since they represent unrecoverable
 * hardware failures such as a dimension mismatch or a kernel launch failure.
 */
struct GpuError {
    std::string message;
    int         code{0}; ///< Vendor error code (cudaError_t, hipError_t, etc.)
};

// ===========================================================================
// Device memory operations.
// CUDA: cudaMalloc/cudaFree/cudaMemcpy
// HIP:  hipMalloc/hipFree/hipMemcpy
// Mock: std::malloc/std::free/std::memcpy
// ===========================================================================

/// Allocate device memory. Returns error on allocation failure.
[[nodiscard]] EINSUMS_EXPORT expected<void *, GpuError> device_malloc(size_t bytes);

/// Free device memory.
EINSUMS_EXPORT void device_free(void *ptr);

/// Copy from host to device.
EINSUMS_EXPORT void memcpy_host_to_device(void *dst, void const *src, size_t bytes);

/// Copy from device to host.
EINSUMS_EXPORT void memcpy_device_to_host(void *dst, void const *src, size_t bytes);

/// Copy from device to device.
EINSUMS_EXPORT void memcpy_device_to_device(void *dst, void const *src, size_t bytes);

/// Set device memory.
EINSUMS_EXPORT void device_memset(void *ptr, int value, size_t bytes);

/// Synchronize the entire device. Blocks until all queued GPU work
/// completes. Wrap GPU calls in this when timing from Python.
APIARY_EXPOSE APIARY_MODULE("gpu") EINSUMS_EXPORT void device_synchronize();

/// Query available (free) device memory in bytes.
/// CUDA/HIP: queries the actual device.
/// Mock: returns a configurable limit (default: system RAM / 2).
APIARY_EXPOSE APIARY_MODULE("gpu") EINSUMS_EXPORT size_t available_device_memory();

/// Set the mock device memory limit (only effective on mock backend).
/// Has no effect when a real GPU is present. This is useful for tests that
/// want to simulate an OOM under the mock.
APIARY_EXPOSE APIARY_MODULE("gpu") EINSUMS_EXPORT void set_mock_device_memory_limit(size_t bytes);

/// Query the device name string.
/// CUDA: cudaGetDeviceProperties().name
/// HIP:  hipGetDeviceProperties().name
/// MPS:  MTLDevice.name
/// Mock: returns ""
APIARY_EXPOSE APIARY_MODULE("gpu") [[nodiscard]] EINSUMS_EXPORT std::string device_name();

// ===========================================================================
// mock-discrete: the device-kernel boundary
// ===========================================================================

#if defined(EINSUMS_HAVE_GPU_MOCK_DISCRETE)

/**
 * @brief Marks a region of code that is standing in for a DEVICE KERNEL.
 *
 * Under mock-discrete, device allocations sit at PROT_NONE so that host code
 * touching a device pointer faults instead of silently succeeding. But the mock
 * backend implements its "device" kernels with host CPU BLAS, and those must be
 * able to read the buffers - on a real GPU, cuBLAS reads device memory as a
 * matter of course. Without this scope the very first mock GEMM segfaults
 * inside OpenBLAS.
 *
 * So the rule the mock enforces is not "nothing may touch device memory", it is
 * "only the backend may". Entering this scope unprotects every live mock
 * allocation; leaving the outermost one re-protects them. Anything outside a
 * scope - most importantly the ComputeGraph executor running a CPU lambda while
 * tensor pointers are still swapped to shadows - still faults.
 *
 * Re-entrant: the templated gpu::blas entry points forward to the typed ones,
 * so scopes nest, and only the outermost re-protects.
 */
struct EINSUMS_EXPORT MockDeviceKernelScope {
    MockDeviceKernelScope();
    ~MockDeviceKernelScope();

    MockDeviceKernelScope(MockDeviceKernelScope const &)            = delete;
    MockDeviceKernelScope &operator=(MockDeviceKernelScope const &) = delete;
};

#    define EINSUMS_GPU_MOCK_KERNEL_SCOPE_CAT2(a, b) a##b
#    define EINSUMS_GPU_MOCK_KERNEL_SCOPE_CAT(a, b)  EINSUMS_GPU_MOCK_KERNEL_SCOPE_CAT2(a, b)
/// Declare a device-kernel scope for the rest of the enclosing block.
#    define EINSUMS_GPU_MOCK_KERNEL_SCOPE                                                                                                  \
        ::einsums::gpu::MockDeviceKernelScope EINSUMS_GPU_MOCK_KERNEL_SCOPE_CAT(_einsums_mock_kernel_scope_, __LINE__)

#else

/// No-op on every backend except mock-discrete.
#    define EINSUMS_GPU_MOCK_KERNEL_SCOPE ((void)0)

#endif

EINSUMS_NAMESPACE_END(gpu)
