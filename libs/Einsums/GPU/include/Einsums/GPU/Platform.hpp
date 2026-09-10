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
// Backend detection: exactly one GPU backend (or mock) is active.
//
// These describe what this translation unit was COMPILED against, not what the
// machine running it has. `has_cuda` means "built with CUDA support", never
// "a usable NVIDIA GPU is present" - a CUDA build on a node with no driver has
// has_cuda == true and no device at all. For the second question, use
// gpu_available() / device_capabilities() below.
//
// The split mirrors libs/Einsums/SIMD/include/Einsums/SIMD/RuntimeFeatures.hpp,
// where compile-time `has_*` constants describe the ISA baseline and
// simd::cpu_features() describes the running machine.
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
/// The plain mock is a CPU-BLAS passthrough over malloc'd "device" memory with
/// unified memory, so it is always numerically correct and never executes the
/// host-to-device transfer path. That makes it useless as a check on the real
/// backends: every CUDA entry point could be a silent no-op - several were -
/// and the whole mock test suite still passes.
///
/// Under mock-discrete the mock instead behaves like a discrete card: no
/// unified memory, device pointers that fault when dereferenced on the host,
/// and the same entry points throwing that throw on CUDA. It needs no GPU, so
/// CI can defend the discrete code paths on any runner.
inline constexpr bool has_mock_discrete =
#if defined(EINSUMS_HAVE_GPU_MOCK_DISCRETE)
    true;
#else
    false;
#endif

/// True if host and device share physical memory (no PCIe copies needed).
/// Apple Silicon (MPS) has unified memory; discrete GPUs (CUDA/HIP) do not.
/// The plain mock backend has no separate device at all, since the host is the
/// device, so it is unified by definition. The Graph executor uses this flag to
/// skip device-shadow allocation and pointer swaps; without is_mock here,
/// the mock path swaps tensor data pointers to uninitialized "shadow"
/// memory and then computes on garbage (e.g. the StridedBatchedGemm
/// GPU-forced test case).
///
/// mock-discrete deliberately opts out, which is the whole point of it: it is
/// what makes the discrete transfer path reachable without hardware.
inline constexpr bool has_unified_memory = has_mps || (is_mock && !has_mock_discrete);

// ---------------------------------------------------------------------------
// Reduced-precision support
//
// These say whether THIS BACKEND IMPLEMENTS the kernel, which is a different
// question from whether the device could run it. Ask device_capabilities() for
// the latter.
// ---------------------------------------------------------------------------

/// True if the active backend implements FP16 GEMM (gpu::blas::hgemm).
///
/// MPS only. The CUDA and HIP hgemm bodies are unimplemented and throw; this
/// used to read `has_cuda || has_mps`, which claimed a kernel that did not
/// exist and, before the throw landed, silently returned an untouched C.
inline constexpr bool has_fp16_gemm = has_mps;

/// True if the active backend implements FP8 E4M3 GEMM (gpu::blas::fp8gemm).
///
/// No backend does yet. This previously tested `__CUDA_ARCH__ >= 890`, which is
/// wrong twice over: __CUDA_ARCH__ is defined only during device compilation,
/// so the constant was false in every host translation unit including on
/// Hopper; and compute capability is a property of the device at run time, not
/// of the build - one binary built with `all-major` targets many architectures.
/// For the hardware question use device_capabilities().fp8_gemm.
inline constexpr bool has_fp8_gemm = false;

// ---------------------------------------------------------------------------
// Runtime device capabilities
// ---------------------------------------------------------------------------

/**
 * @brief What the GPU on THIS machine can actually do, detected at run time.
 *
 * The runtime counterpart to the compile-time `has_*` constants above: the same
 * vocabulary, but describing the device the process found rather than the
 * backend it was built against.
 *
 * On the mock backends the host is the device, so `available` is true and
 * `device_count` is 1; the arithmetic fields are false because the mock
 * implements no reduced-precision kernels.
 */
struct DeviceCapabilities {
    /// A device is present AND usable: driver loaded, at least one device
    /// enumerated, and a context creatable. False on a CUDA-enabled build
    /// running on a machine with no GPU or a broken driver.
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
 * Runs once on first call (thread-safe) and caches the result, because probing
 * creates a CUDA context and that is far too expensive to repeat per query.
 *
 * Never throws: a missing driver or a failed enumeration yields a
 * default-constructed value with `available == false`, so a CUDA-enabled binary
 * stays usable on a machine without a GPU.
 */
[[nodiscard]] EINSUMS_EXPORT DeviceCapabilities const &device_capabilities();

/**
 * @brief True when GPU work can actually run right now.
 *
 * This is the check that gates offload, not `has_gpu`. `has_gpu` is a build
 * flag and cannot tell a working card from an absent one, which is why a
 * CUDA-built binary on a driverless node used to allocate a null device shadow,
 * swap a tensor's data pointer to it, and segfault in the CPU fallback.
 */
[[nodiscard]] EINSUMS_EXPORT bool gpu_available();

EINSUMS_NAMESPACE_END(gpu)
