//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// This header is included from EinsumPackedGemm.hpp.
//
// Public entry point for the BLIS-style register-blocked micro-kernel that
// replaces the per-tile vendor blas::gemm calls (the stand-in left behind
// when the MLIR-JIT micro-kernel was removed). The kernel bodies live in
// MicroKernelBody.hpp and are compiled once per SIMD-dispatch rung by
// einsums_add_simd_dispatch_sources() (see src/MicroKernelImpl.cpp); the
// rung is resolved at runtime in src/MicroKernelDispatch.cpp, following the
// same pattern HPTT uses for its transpose kernels. Without this, the MR=8
// (AVX) and MR=16 (AVX-512) block shapes chosen by cpu_config() would be
// compiled at the baseline ISA — an 8x6 double block needs 24 accumulator
// registers, which baseline SSE2's 16 half-width xmm registers cannot hold
// without spilling in the innermost FMA loop.

#include <Einsums/Config.hpp>

#include <Einsums/PackedGemm/Packing.hpp>

#define EINSUMS_PACKED_GEMM_KERNEL_NS arch_ambient
#include <Einsums/PackedGemm/MicroKernelBody.hpp>

#include <complex>
#include <cstdint>

namespace einsums::packed_gemm {

/// @brief Register-block shape (MR, NR) the resolved micro-kernel wants.
///
/// The kernel rung owns the packing geometry: NEON/AVX rungs use
/// cpu_config()'s vector blocking (MR = 2*VL, NR = 6), while the SME rung
/// uses ZA-tile blocking (MR = NR = 2 * streaming-VL doubles, 16x16 on
/// Apple M4). blis_contraction must pack panels with the shape of the
/// kernel it resolves, so the two are queried together.
struct MicroKernelShape {
    int mr;
    int nr;
};

/// @brief Signature of a resolved micro-kernel tile function.
///
/// Arguments: (mr_block, nr_block, kc, alpha, Ap_panel, Bp_panel, mr_eff,
/// nr_eff, C, rs_c, cs_c) — see MicroKernelBody.hpp for semantics.
template <typename T>
using MicroKernelFn = void (*)(int, int, int64_t, T, T const *, T const *, int64_t, int64_t, T *, int64_t, int64_t);

namespace detail {

/// Ambient-flags fallback used for element types without a per-rung build.
template <typename T>
void micro_kernel_ambient(int mr_block, int nr_block, int64_t kc, T alpha, T const *Ap, T const *Bp, int64_t mr_eff, int64_t nr_eff, T *C,
                          int64_t rs_c, int64_t cs_c) {
    arch_ambient::micro_kernel_run<T>(mr_block, nr_block, kc, alpha, Ap, Bp, mr_eff, nr_eff, C, rs_c, cs_c);
}

} // namespace detail

/// @brief Resolve the micro-kernel for element type T.
///
/// For float, double, and their complex forms the specializations (defined in
/// src/MicroKernelDispatch.cpp) return the entry of the highest SIMD-dispatch
/// rung the CPU supports; the result is cached, so callers should still hoist
/// the call out of tile loops. Any other element type gets the ambient-flags
/// header instantiation.
template <typename T>
MicroKernelFn<T> micro_kernel_entry() {
    return &detail::micro_kernel_ambient<T>;
}

template <>
EINSUMS_EXPORT MicroKernelFn<float> micro_kernel_entry<float>();
template <>
EINSUMS_EXPORT MicroKernelFn<double> micro_kernel_entry<double>();
template <>
EINSUMS_EXPORT MicroKernelFn<std::complex<float>> micro_kernel_entry<std::complex<float>>();
template <>
EINSUMS_EXPORT MicroKernelFn<std::complex<double>> micro_kernel_entry<std::complex<double>>();

/// @brief The register-block shape of the kernel micro_kernel_entry<T>()
///        resolves to. Query both together and pack panels with this shape.
template <typename T>
MicroKernelShape micro_kernel_shape() {
    auto const &cfg = cpu_config();
    return {cfg.MR, cfg.NR};
}

template <>
EINSUMS_EXPORT MicroKernelShape micro_kernel_shape<float>();
template <>
EINSUMS_EXPORT MicroKernelShape micro_kernel_shape<double>();
template <>
EINSUMS_EXPORT MicroKernelShape micro_kernel_shape<std::complex<float>>();
template <>
EINSUMS_EXPORT MicroKernelShape micro_kernel_shape<std::complex<double>>();

} // namespace einsums::packed_gemm
