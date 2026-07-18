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

#include <Einsums/PackedGemm/MicroKernel.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <complex>
#include <cstdint>

namespace einsums::packed_gemm {

#define EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES(ns)                                                                                       \
    namespace ns {                                                                                                                         \
    template <typename T>                                                                                                                  \
    void micro_kernel_tile(int mr_block, int nr_block, int64_t kc, T alpha, T const *Ap, T const *Bp, int64_t mr_eff, int64_t nr_eff,      \
                           T *C, int64_t rs_c, int64_t cs_c);                                                                              \
    template <typename T>                                                                                                                  \
    MicroKernelShape micro_kernel_block();                                                                                                 \
    }

#if defined(EINSUMS_SIMD_HAS_RUNG_NATIVE)
EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES(arch_native)
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_SME)
EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES(arch_sme)
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_BASELINE)
EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES(arch_baseline)
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_V2)
EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES(arch_v2)
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_V3)
EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES(arch_v3)
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_V4)
EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES(arch_v4)
#endif

#undef EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES

namespace {

// Ladder slot macros: nullptr keeps the position of rungs that were not
// built. `arch_native` (single-TU aarch64/pinned builds) occupies the
// baseline slot - on aarch64 it IS the toolchain baseline (NEON) - so the
// sme slot can still out-rank it through the normal ladder walk.
#if defined(EINSUMS_SIMD_HAS_RUNG_NATIVE)
#    define EINSUMS_PACKED_GEMM_BASELINE_ENTRY(ns_fn) &arch_native::ns_fn
#elif defined(EINSUMS_SIMD_HAS_RUNG_BASELINE)
#    define EINSUMS_PACKED_GEMM_BASELINE_ENTRY(ns_fn) &arch_baseline::ns_fn
#else
#    define EINSUMS_PACKED_GEMM_BASELINE_ENTRY(ns_fn) nullptr
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_V2)
#    define EINSUMS_PACKED_GEMM_V2_ENTRY(ns_fn) &arch_v2::ns_fn
#else
#    define EINSUMS_PACKED_GEMM_V2_ENTRY(ns_fn) nullptr
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_V3)
#    define EINSUMS_PACKED_GEMM_V3_ENTRY(ns_fn) &arch_v3::ns_fn
#else
#    define EINSUMS_PACKED_GEMM_V3_ENTRY(ns_fn) nullptr
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_V4)
#    define EINSUMS_PACKED_GEMM_V4_ENTRY(ns_fn) &arch_v4::ns_fn
#else
#    define EINSUMS_PACKED_GEMM_V4_ENTRY(ns_fn) nullptr
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_SME)
#    define EINSUMS_PACKED_GEMM_SME_ENTRY(ns_fn) &arch_sme::ns_fn
#else
#    define EINSUMS_PACKED_GEMM_SME_ENTRY(ns_fn) nullptr
#endif

#define EINSUMS_PACKED_GEMM_LADDER(ns_fn)                                                                                                  \
    EINSUMS_PACKED_GEMM_BASELINE_ENTRY(ns_fn), EINSUMS_PACKED_GEMM_V2_ENTRY(ns_fn), EINSUMS_PACKED_GEMM_V3_ENTRY(ns_fn),                   \
        EINSUMS_PACKED_GEMM_V4_ENTRY(ns_fn), EINSUMS_PACKED_GEMM_SME_ENTRY(ns_fn)

} // namespace

#define EINSUMS_PACKED_GEMM_DEFINE_ENTRY(T)                                                                                                \
    template <>                                                                                                                            \
    EINSUMS_EXPORT MicroKernelFn<T> micro_kernel_entry<T>() {                                                                              \
        static MicroKernelFn<T> const fn = einsums::simd::select<MicroKernelFn<T>>(EINSUMS_PACKED_GEMM_LADDER(micro_kernel_tile<T>));      \
        return fn;                                                                                                                         \
    }                                                                                                                                      \
    template <>                                                                                                                            \
    EINSUMS_EXPORT MicroKernelShape micro_kernel_shape<T>() {                                                                              \
        using ShapeFn                       = MicroKernelShape (*)();                                                                      \
        static ShapeFn const          fn    = einsums::simd::select<ShapeFn>(EINSUMS_PACKED_GEMM_LADDER(micro_kernel_block<T>));           \
        static MicroKernelShape const shape = fn();                                                                                        \
        return shape;                                                                                                                      \
    }

EINSUMS_PACKED_GEMM_DEFINE_ENTRY(float)
EINSUMS_PACKED_GEMM_DEFINE_ENTRY(double)
EINSUMS_PACKED_GEMM_DEFINE_ENTRY(std::complex<float>)
EINSUMS_PACKED_GEMM_DEFINE_ENTRY(std::complex<double>)

#undef EINSUMS_PACKED_GEMM_DEFINE_ENTRY
#undef EINSUMS_PACKED_GEMM_LADDER

} // namespace einsums::packed_gemm
