//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <string>

EINSUMS_NAMESPACE_BEGIN(gpu)

// ---------------------------------------------------------------------------
// Backend detection: exactly one GPU backend (or mock) is active. These describe the build, not the
// machine: for a usable device, ask gpu_available() / device_capabilities().
// ---------------------------------------------------------------------------

inline constexpr bool has_cuda =
#if defined(EINSUMS_HAVE_CUDA)
    true;
#else
    false;
#endif

inline constexpr bool has_hip =
#if defined(EINSUMS_HAVE_HIP)
    true;
#else
    false;
#endif

inline constexpr bool has_mps =
#if defined(EINSUMS_HAVE_MPS)
    true;
#else
    false;
#endif

/// True if any real GPU backend is available.
inline constexpr bool has_gpu = has_cuda || has_hip || has_mps;

/// True if running with the mock CPU backend, meaning no real GPU.
inline constexpr bool is_mock = !has_gpu;

/// True when the mock backend is configured to imitate a DISCRETE device
/// (EINSUMS_WITH_GPU_MOCK_DISCRETE=ON).
///
/// The plain mock never runs the transfer path. Mock-discrete behaves like a discrete card (no
/// unified memory, device pointers fault on the host), so CI exercises those paths without a GPU.
inline constexpr bool has_mock_discrete =
#if defined(EINSUMS_HAVE_GPU_MOCK_DISCRETE)
    true;
#else
    false;
#endif

/// True if host and device share memory: MPS and the plain mock, not CUDA/HIP or mock-discrete.
/// The executor then skips device shadows.
inline constexpr bool has_unified_memory = has_mps || (is_mock && !has_mock_discrete);

// ---------------------------------------------------------------------------
// Reduced-precision support
//
// Whether the backend implements the kernel; device_capabilities() says whether the device could.
// ---------------------------------------------------------------------------

/// True if the active backend implements FP16 GEMM (gpu::blas::hgemm).
///
/// MPS only; the CUDA and HIP bodies throw.
inline constexpr bool has_fp16_gemm = has_mps;

/// True if the active backend implements FP8 E4M3 GEMM (gpu::blas::fp8gemm).
///
/// No backend does yet. Whether the hardware could is device_capabilities().fp8_gemm.
inline constexpr bool has_fp8_gemm = false;

// ---------------------------------------------------------------------------
// Runtime device capabilities
// ---------------------------------------------------------------------------

/**
 * @brief What the GPU on THIS machine can actually do, detected at run time.
 *
 * The runtime counterpart of the `has_*` constants. On the mocks `available` is true and
 * `device_count` 1. Not named `DeviceCapabilities`, which winspool.h defines as a macro.
 */
struct DeviceCaps {
    /// A device is present and usable: driver loaded, a device enumerated, a context creatable.
    bool available{false};

    /// Number of devices visible to this process (honors CUDA_VISIBLE_DEVICES).
    int device_count{0};

    /// Compute capability of device 0. Zero when `available` is false.
    int compute_major{0};
    int compute_minor{0};

    /// Device-reported unified/managed addressing. Distinct from the
    /// compile-time has_unified_memory, which is what the executor branches on.
    bool unified_memory{false};

    bool fp16_gemm{false}; ///< Tensor-core FP16 (CUDA sm_70+).
    bool bf16_gemm{false}; ///< BFloat16 (CUDA sm_80+). False on Turing.
    bool fp8_gemm{false};  ///< FP8 E4M3 (CUDA sm_89+). False on Turing.

    std::size_t total_memory{0}; ///< Total device memory in bytes.
    std::string name;            ///< Device name, empty when unavailable.
};

/**
 * @brief Detect the capabilities of the GPU this process can see.
 *
 * Probed once (it creates a CUDA context) and cached. Never throws: without a driver,
 * `available == false`.
 */
[[nodiscard]] EINSUMS_EXPORT DeviceCaps const &device_capabilities();

/**
 * @brief True when GPU work can actually run right now.
 *
 * What gates offload; `has_gpu` is only a build flag.
 */
[[nodiscard]] EINSUMS_EXPORT bool gpu_available();

EINSUMS_NAMESPACE_END(gpu)
