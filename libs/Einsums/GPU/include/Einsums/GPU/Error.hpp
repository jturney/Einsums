//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPU/Platform.hpp>

#include <source_location>
#include <stdexcept>
#include <string>

// Include vendor error headers
#if defined(EINSUMS_HAVE_CUDA)
#    include <cublas_v2.h>
#    include <cuda_runtime_api.h>
#    include <cusolverDn.h>
#elif defined(EINSUMS_HAVE_HIP)
#    include <hip/hip_runtime_api.h>
#    include <hipblas/hipblas.h>
#    include <hipsolver/hipsolver.h>
#endif

EINSUMS_NAMESPACE_BEGIN(gpu)

/**
 * @brief Report a GPU entry point that has no implementation on this backend.
 *
 * Several vendor entry points are declared and dispatched to but not yet
 * written for CUDA/HIP. They used to cast their arguments to void and return,
 * which is the worst available behavior: the caller sees a successful call and
 * an untouched output buffer, so a missing kernel is indistinguishable from a
 * correct one that happened to compute zeros. Under the mock backend the same
 * functions delegate to CPU BLAS and are numerically right, so nothing in the
 * test suite noticed.
 *
 * Throwing instead makes the gap loud at the call site. Callers that can fall
 * back (the ComputeGraph executor catches around its GPU dispatch) still do so;
 * callers that cannot get a diagnosable error instead of silent corruption.
 *
 * @param what Name of the entry point, e.g. "gpu::solver::syev<float>".
 */
[[noreturn]] inline void not_implemented(char const                *what,
                                         std::source_location const loc = std::source_location::current()) {
    EINSUMS_THROW_EXCEPTION(std::runtime_error, "{} is not implemented for this GPU backend (at {}:{})", what, loc.file_name(), loc.line());
    // EINSUMS_THROW_EXCEPTION always throws; this satisfies [[noreturn]] for
    // compilers that cannot see through the macro.
    throw std::runtime_error(what);
}

// ===========================================================================
// Unified error checking macros for GPU runtime, BLAS, and solver calls.
//
// Usage:
//   gpu_catch(cudaMalloc(&ptr, bytes));
//   gpu_blas_catch(cublasSgemm(...));
//   gpu_solver_catch(cusolverDnDsyev(...));
//
// On the mock backend, these are no-ops since there are no GPU errors.
// ===========================================================================

#if defined(EINSUMS_HAVE_CUDA)

namespace detail {
inline void check_cuda_error(cudaError_t err, std::source_location loc = std::source_location::current()) {
    if (err != cudaSuccess) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "CUDA error at {}:{}: {} ({})", loc.file_name(), loc.line(), cudaGetErrorString(err),
                                static_cast<int>(err));
    }
}

inline void check_cublas_error(cublasStatus_t err, std::source_location loc = std::source_location::current()) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "cuBLAS error at {}:{}: status {}", loc.file_name(), loc.line(), static_cast<int>(err));
    }
}

inline void check_cusolver_error(cusolverStatus_t err, std::source_location loc = std::source_location::current()) {
    if (err != CUSOLVER_STATUS_SUCCESS) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "cuSOLVER error at {}:{}: status {}", loc.file_name(), loc.line(),
                                static_cast<int>(err));
    }
}
} // namespace detail

#    define gpu_catch(call)        ::einsums::gpu::detail::check_cuda_error(call)
#    define gpu_blas_catch(call)   ::einsums::gpu::detail::check_cublas_error(call)
#    define gpu_solver_catch(call) ::einsums::gpu::detail::check_cusolver_error(call)

#elif defined(EINSUMS_HAVE_HIP)

namespace detail {
inline void check_hip_error(hipError_t err, std::source_location loc = std::source_location::current()) {
    if (err != hipSuccess) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "HIP error at {}:{}: {} ({})", loc.file_name(), loc.line(), hipGetErrorString(err),
                                static_cast<int>(err));
    }
}

inline void check_hipblas_error(hipblasStatus_t err, std::source_location loc = std::source_location::current()) {
    if (err != HIPBLAS_STATUS_SUCCESS) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "hipBLAS error at {}:{}: status {}", loc.file_name(), loc.line(),
                                static_cast<int>(err));
    }
}

inline void check_hipsolver_error(hipsolverStatus_t err, std::source_location loc = std::source_location::current()) {
    if (err != HIPSOLVER_STATUS_SUCCESS) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "hipSOLVER error at {}:{}: status {}", loc.file_name(), loc.line(),
                                static_cast<int>(err));
    }
}
} // namespace detail

#    define gpu_catch(call)        ::einsums::gpu::detail::check_hip_error(call)
#    define gpu_blas_catch(call)   ::einsums::gpu::detail::check_hipblas_error(call)
#    define gpu_solver_catch(call) ::einsums::gpu::detail::check_hipsolver_error(call)

#else // Mock backend: no GPU errors possible

#    define gpu_catch(call)        ((void)(call))
#    define gpu_blas_catch(call)   ((void)(call))
#    define gpu_solver_catch(call) ((void)(call))

#endif

EINSUMS_NAMESPACE_END(gpu)
