//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Arch-neutral micro-kernel dispatch: the one place the SIMD rung is chosen
// for the packed-GEMM tile kernel. Mirrors HPTT's TransposeFactory.cpp.
//
// The kernel implementation (MicroKernelImpl.cpp) is compiled once per
// instruction-set rung by einsums_add_simd_dispatch_sources(), each copy in
// its own namespace (packed_gemm::arch_baseline, packed_gemm::arch_v3, ...,
// and on aarch64 arch_native plus optionally arch_sme). This TU is compiled
// exactly once, WITHOUT arch flags: it declares each rung's entry points
// (guarded by the EINSUMS_SIMD_HAS_RUNG_* definitions the CMake helper
// emits) and picks the best one at or below einsums::simd::selected_arch(),
// cached per element type. The kernel and its block shape resolve through
// the same ladder so packing geometry always matches the kernel.

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/PackedGemm/MicroKernel.hpp>
#include <Einsums/SIMD/RungLadder.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <complex>
#include <cstdint>

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

#define EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES(ns)                                                                                       \
    namespace ns {                                                                                                                         \
    template <typename T>                                                                                                                  \
    void micro_kernel_tile(int mr_block, int nr_block, int64_t kc, T alpha, T const *Ap, T const *Bp, int64_t mr_eff, int64_t nr_eff,      \
                           T *C, int64_t rs_c, int64_t cs_c);                                                                              \
    template <typename T>                                                                                                                  \
    MicroKernelShape micro_kernel_block();                                                                                                 \
    }

EINSUMS_SIMD_FOR_EACH_BUILT_RUNG(EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES)

#undef EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES

#define EINSUMS_PACKED_GEMM_DEFINE_ENTRY(T)                                                                                                \
    template <>                                                                                                                            \
    EINSUMS_EXPORT MicroKernelFn<T> micro_kernel_entry<T>() {                                                                              \
        static MicroKernelFn<T> const fn = einsums::simd::select<MicroKernelFn<T>>(EINSUMS_SIMD_LADDER(micro_kernel_tile<T>));             \
        return fn;                                                                                                                         \
    }                                                                                                                                      \
    template <>                                                                                                                            \
    EINSUMS_EXPORT MicroKernelShape micro_kernel_shape<T>() {                                                                              \
        using ShapeFn                       = MicroKernelShape (*)();                                                                      \
        static ShapeFn const          fn    = einsums::simd::select<ShapeFn>(EINSUMS_SIMD_LADDER(micro_kernel_block<T>));                  \
        static MicroKernelShape const shape = [] {                                                                                         \
            MicroKernelShape const s = fn();                                                                                               \
            EINSUMS_LOG_INFO("packed_gemm kernel<{}>: rung={}, tile MR={} x NR={}, kc_hint={}, block_gemm={}, fast_scatter={}", #T,        \
                             einsums::simd::to_string(einsums::simd::selected_arch()), s.mr, s.nr, s.kc, s.block_gemm, s.fast_scatter);    \
            return s;                                                                                                                      \
        }();                                                                                                                               \
        return shape;                                                                                                                      \
    }

EINSUMS_PACKED_GEMM_DEFINE_ENTRY(float)
EINSUMS_PACKED_GEMM_DEFINE_ENTRY(double)
EINSUMS_PACKED_GEMM_DEFINE_ENTRY(std::complex<float>)
EINSUMS_PACKED_GEMM_DEFINE_ENTRY(std::complex<double>)

#undef EINSUMS_PACKED_GEMM_DEFINE_ENTRY

EINSUMS_NAMESPACE_END(packed_gemm)
