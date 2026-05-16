//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

namespace einsums::gpu {

// ---------------------------------------------------------------------------
// Backend detection — exactly one GPU backend (or mock) is active.
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

/// True if running with the mock (CPU) backend — no real GPU.
inline constexpr bool is_mock = !has_gpu;

// ---------------------------------------------------------------------------
// Reduced-precision feature detection
// ---------------------------------------------------------------------------

/// True if host and device share physical memory (no PCIe copies needed).
/// Apple Silicon has unified memory. Discrete GPUs (CUDA/HIP) do not.
inline constexpr bool has_unified_memory = has_mps;

/// FP16 GEMM: available on CUDA tensor-core GPUs (Volta+) and MPS (Apple Silicon).
inline constexpr bool has_fp16_gemm = has_cuda || has_mps;

/// FP8 E4M3 tensor core GEMM requires Hopper (sm_89+) or Blackwell.
inline constexpr bool has_fp8_gemm =
#if defined(EINSUMS_HAVE_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    true;
#else
    false;
#endif

} // namespace einsums::gpu
