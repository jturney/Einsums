//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/GPU/Error.hpp>
#include <Einsums/GPU/Runtime.hpp>

#if defined(EINSUMS_HAVE_CUDA)
#    include <cuda_runtime_api.h>
#elif defined(EINSUMS_HAVE_HIP)
#    include <hip/hip_runtime_api.h>
#elif defined(EINSUMS_HAVE_MPS)
#    include <Einsums/GPU/MPSBackend.hpp>
#else
#    include <cstdlib>
#    include <cstring>
#    include <unistd.h>
#endif

#include <atomic>
#include <cstring>

namespace einsums::gpu {

expected<void *, GpuError> device_malloc(size_t bytes) {
    void *ptr = nullptr;
#if defined(EINSUMS_HAVE_CUDA)
    auto err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess)
        return unexpected(GpuError{fmt::format("cudaMalloc({} bytes) failed: {}", bytes, cudaGetErrorString(err)), static_cast<int>(err)});
#elif defined(EINSUMS_HAVE_HIP)
    auto err = hipMalloc(&ptr, bytes);
    if (err != hipSuccess)
        return unexpected(GpuError{fmt::format("hipMalloc({} bytes) failed: {}", bytes, hipGetErrorString(err)), static_cast<int>(err)});
#elif defined(EINSUMS_HAVE_MPS)
    ptr = mps::device_malloc(bytes);
    if (!ptr && bytes > 0)
        return unexpected(GpuError{.message = fmt::format("MPS device_malloc({} bytes) failed", bytes), .code = -1});
#else
    ptr = std::malloc(bytes);
    if (!ptr && bytes > 0)
        return unexpected(GpuError{fmt::format("malloc({} bytes) failed", bytes), -1});
#endif
    return ptr;
}

void device_free(void *ptr) {
    if (ptr == nullptr)
        return;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaFree(ptr));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipFree(ptr));
#elif defined(EINSUMS_HAVE_MPS)
    mps::device_free(ptr);
#else
    std::free(ptr);
#endif
}

void memcpy_host_to_device(void *dst, void const *src, size_t bytes) {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
#elif defined(EINSUMS_HAVE_MPS)
    mps::memcpy_host_to_device(dst, src, bytes);
#else
    std::memcpy(dst, src, bytes);
#endif
}

void memcpy_device_to_host(void *dst, void const *src, size_t bytes) {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
#elif defined(EINSUMS_HAVE_MPS)
    mps::memcpy_device_to_host(dst, src, bytes);
#else
    std::memcpy(dst, src, bytes);
#endif
}

void memcpy_device_to_device(void *dst, void const *src, size_t bytes) {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
#elif defined(EINSUMS_HAVE_MPS)
    mps::memcpy_device_to_device(dst, src, bytes);
#else
    std::memcpy(dst, src, bytes);
#endif
}

void device_memset(void *ptr, int value, size_t bytes) {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaMemset(ptr, value, bytes));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipMemset(ptr, value, bytes));
#elif defined(EINSUMS_HAVE_MPS)
    mps::device_memset(ptr, value, bytes);
#else
    std::memset(ptr, value, bytes);
#endif
}

void device_synchronize() {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaDeviceSynchronize());
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipDeviceSynchronize());
#elif defined(EINSUMS_HAVE_MPS)
    mps::device_synchronize();
#else
    // Mock: everything is synchronous, nothing to wait for.
#endif
}

namespace {

/// Override for available_device_memory (0 = use real value).
/// Works on all backends for testing budget-constrained placement.
std::atomic<size_t> g_memory_override{0};

#if !defined(EINSUMS_HAVE_CUDA) && !defined(EINSUMS_HAVE_HIP) && !defined(EINSUMS_HAVE_MPS)
/// Mock device memory limit (0 = use default: system RAM / 2).
std::atomic<size_t> mock_memory_limit{0};

size_t default_mock_limit() {
#    if defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    auto pages     = sysconf(_SC_PHYS_PAGES);
    auto page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<size_t>(pages) * static_cast<size_t>(page_size) / 2;
    }
#    endif
    return size_t{4} * 1024 * 1024 * 1024; // 4 GB fallback
}
#endif

} // namespace

size_t available_device_memory() {
    // Check test override first (works on all backends).
    size_t override_val = g_memory_override.load(std::memory_order_relaxed);
    if (override_val > 0)
        return override_val;

#if defined(EINSUMS_HAVE_CUDA)
    size_t free_mem = 0, total_mem = 0;
    gpu_catch(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
#elif defined(EINSUMS_HAVE_HIP)
    size_t free_mem = 0, total_mem = 0;
    gpu_catch(hipMemGetInfo(&free_mem, &total_mem));
    return free_mem;
#elif defined(EINSUMS_HAVE_MPS)
    return mps::available_device_memory();
#else
    return default_mock_limit();
#endif
}

void set_mock_device_memory_limit(size_t bytes) {
    g_memory_override.store(bytes, std::memory_order_relaxed);
}

std::string device_name() {
#if defined(EINSUMS_HAVE_CUDA)
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess) {
        return std::string(props.name);
    }
    return "Unknown CUDA Device";
#elif defined(EINSUMS_HAVE_HIP)
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
        return std::string(props.name);
    }
    return "Unknown HIP Device";
#elif defined(EINSUMS_HAVE_MPS)
    return mps::device_name();
#else
    return "";
#endif
}

} // namespace einsums::gpu
