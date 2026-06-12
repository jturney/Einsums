//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <string>

namespace einsums::gpu {

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

} // namespace einsums::gpu
