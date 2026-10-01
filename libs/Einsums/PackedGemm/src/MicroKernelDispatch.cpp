//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Arch-neutral micro-kernel dispatch: the one place the SIMD rung is chosen
// for the packed-GEMM tile kernel. Mirrors HPTT's TransposeFactory.cpp.
//
// The kernel implementation (MicroKernelImpl.cpp) is compiled once per
// instruction-set rung by stripes_add_dispatch_sources(), each copy in
// its own namespace (packed_gemm::arch_baseline, packed_gemm::arch_v3, ...,
// and on aarch64 arch_native plus optionally arch_sme). This TU is compiled
// exactly once, WITHOUT arch flags: it declares each rung's entry points
// (guarded by the STRIPES_HAS_RUNG_* definitions the CMake helper
// emits) and picks the best built one the machine supports, starting from
// stripes::selected_arch(), cached per element type. The kernel and its block shape resolve through
// the same ladder so packing geometry always matches the kernel.

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/PackedGemm/MicroKernel.hpp>
#include <Einsums/PackedGemm/Options.hpp>

#include <Stripes/RungLadder.hpp>
#include <Stripes/RuntimeFeatures.hpp>
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
    template <typename T>                                                                                                                  \
    MicroKernelShape micro_kernel_block_1m();                                                                                              \
    template <typename T>                                                                                                                  \
    void pack_transpose_rows(T *panel, T const *const *rows, int64_t nrows, int64_t kc, int64_t ld);                                       \
    }

STRIPES_FOR_EACH_BUILT_RUNG(EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES)

#undef EINSUMS_PACKED_GEMM_DECLARE_RUNG_ENTRIES

namespace {

/// The shape the selected rung's 1m route advertises for complex @p T (spelled
/// @p type_name in the log), which is
/// the rung's ordinary shape where it has no 1m route to opt into.
///
/// Resolved once, like the default shape above, but read only while the flag is
/// set: the flag is read on every call so a test or a caller that sets it at run
/// time sees the change on its next contraction.
template <typename T>
MicroKernelShape complex_1m_shape(char const *type_name) {
    using ShapeFn                       = MicroKernelShape (*)();
    static ShapeFn const          fn    = stripes::select<ShapeFn>(STRIPES_LADDER(micro_kernel_block_1m<T>));
    static MicroKernelShape const shape = [type_name] {
        MicroKernelShape const s = fn();
        EINSUMS_LOG_INFO("packed_gemm kernel<{}> with --{}: rung={}, tile MR={} x NR={}, use_1m={}", type_name,
                         option::PackedGemmComplex1m.name, stripes::to_string(stripes::selected_arch()), s.mr, s.nr, s.use_1m);
        return s;
    }();
    return shape;
}

} // namespace

#define EINSUMS_PACKED_GEMM_DEFINE_ENTRY(T)                                                                                                \
    template <>                                                                                                                            \
    EINSUMS_EXPORT MicroKernelFn<T> micro_kernel_entry<T>() {                                                                              \
        static MicroKernelFn<T> const fn = stripes::select<MicroKernelFn<T>>(STRIPES_LADDER(micro_kernel_tile<T>));                        \
        return fn;                                                                                                                         \
    }                                                                                                                                      \
    template <>                                                                                                                            \
    EINSUMS_EXPORT MicroKernelShape micro_kernel_shape<T>() {                                                                              \
        using ShapeFn                       = MicroKernelShape (*)();                                                                      \
        static ShapeFn const          fn    = stripes::select<ShapeFn>(STRIPES_LADDER(micro_kernel_block<T>));                             \
        static MicroKernelShape const shape = [] {                                                                                         \
            MicroKernelShape const s = fn();                                                                                               \
            EINSUMS_LOG_INFO("packed_gemm kernel<{}>: rung={}, tile MR={} x NR={}, kc_hint={}, block_gemm={}, fast_scatter={}", #T,        \
                             stripes::to_string(stripes::selected_arch()), s.mr, s.nr, s.kc, s.block_gemm, s.fast_scatter);                \
            return s;                                                                                                                      \
        }();                                                                                                                               \
        if constexpr (IsComplexV<T>) {                                                                                                     \
            if (config::get(option::PackedGemmComplex1m)) {                                                                                \
                return complex_1m_shape<T>(#T);                                                                                            \
            }                                                                                                                              \
        }                                                                                                                                  \
        return shape;                                                                                                                      \
    }

#define EINSUMS_PACKED_GEMM_DEFINE_PACK_ENTRY(T)                                                                                           \
    template <>                                                                                                                            \
    EINSUMS_EXPORT PackTransposeFn<T> pack_transpose_entry<T>() {                                                                          \
        static PackTransposeFn<T> const fn = stripes::select<PackTransposeFn<T>>(STRIPES_LADDER(pack_transpose_rows<T>));                  \
        return fn;                                                                                                                         \
    }

EINSUMS_PACKED_GEMM_DEFINE_PACK_ENTRY(float)
EINSUMS_PACKED_GEMM_DEFINE_PACK_ENTRY(double)

#undef EINSUMS_PACKED_GEMM_DEFINE_PACK_ENTRY

EINSUMS_PACKED_GEMM_DEFINE_ENTRY(float)
EINSUMS_PACKED_GEMM_DEFINE_ENTRY(double)
EINSUMS_PACKED_GEMM_DEFINE_ENTRY(std::complex<float>)
EINSUMS_PACKED_GEMM_DEFINE_ENTRY(std::complex<double>)

#undef EINSUMS_PACKED_GEMM_DEFINE_ENTRY

EINSUMS_NAMESPACE_END(packed_gemm)
