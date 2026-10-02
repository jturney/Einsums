//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// This header is included from PackedGemm.hpp.
//
// The BLIS-style register-blocked micro-kernel. Bodies in MicroKernelBody.hpp are compiled per
// rung (src/MicroKernelImpl.cpp) and resolved at run time (src/MicroKernelDispatch.cpp), as HPTT's
// are, so each block shape gets the registers it needs.

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/Packing.hpp>

#define EINSUMS_PACKED_GEMM_KERNEL_NS arch_ambient
#include <Einsums/PackedGemm/MicroKernelBody.hpp>

#include <complex>
#include <cstdint>

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

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
    /// K-block hint: 0 = use the cache-derived default (compute_blocking).
    ///
    /// SME raises it, as ZA holds C across the K loop; its M block shrinks to keep A in L2.
    int64_t kc = 0;
    /// The scatter path beats Sort+GEMM on this rung (2.3x on M4 SME), so callers with that fallback
    /// still take it.
    bool fast_scatter = false;
    /// Scatter engine: a vendor GEMM per cache block, then scatter (for vendors reaching matrix
    /// units, e.g. AMX on M1-M3), or false for the rung's own tile kernel.
    bool block_gemm = true;
    /// Complex elements via Van Zee's 1m method: A packs in the expanded 1e
    /// form ([[ar,-ai],[ai,ar]] per element), B packs re/im as adjacent K
    /// rows (1r), and the REAL tile kernel of the underlying real type
    /// computes interleaved-complex output directly.
    ///
    /// mr/nr/kc then describe the real kernel, and the extents double (Mh = 2M, Kh = 2K). 1.74x over
    /// Sort+GEMM for complex<double> on M4 SME.
    bool use_1m = false;
    /// Complex elements via the 3m (Karatsuba) method on the BLOCK-GEMM
    /// path: three real GEMMs per block, 25% fewer flops and able to reach AMX. Slightly weaker error
    /// bounds (Higham); enable only where measured.
    bool use_3m = false;
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

EINSUMS_NAMESPACE_END(packed_gemm)
