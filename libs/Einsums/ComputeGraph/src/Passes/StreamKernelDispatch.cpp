//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Arch-neutral dispatch for the stream-fusion inner kernel: the one place the
// SIMD rung is chosen. StreamKernelImpl.cpp is compiled once per rung by
// einsums_add_simd_dispatch_sources() (each copy in its own namespace,
// arch_baseline/arch_v2/arch_v3/arch_v4, or arch_native on aarch64/pinned
// builds). This TU is compiled exactly once, WITHOUT arch flags: it declares
// each rung's entry (guarded by the EINSUMS_SIMD_HAS_RUNG_* definitions the
// CMake helper emits) and picks the best one at or below
// einsums::simd::selected_arch(), cached per element type. No `sme` rung is
// built for this kernel - a bandwidth-bound streaming FMA gains nothing from
// the matrix unit - so the select ladder leaves that slot at its default.

#include <Einsums/ComputeGraph/Passes/StreamKernel.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <complex>
#include <cstdint>

namespace einsums::compute_graph::passes {

#define EINSUMS_STREAM_DECLARE_RUNG_ENTRY(ns)                                                                                              \
    namespace ns {                                                                                                                         \
    template <typename T>                                                                                                                  \
    void stream_inner(T *cb, T const *sp, T const *w, T alpha, int64_t n, int64_t co, int64_t si, int64_t wo, int64_t ds, int64_t dc,      \
                      int64_t dw);                                                                                                         \
    }

#if defined(EINSUMS_SIMD_HAS_RUNG_NATIVE)
EINSUMS_STREAM_DECLARE_RUNG_ENTRY(arch_native)
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_BASELINE)
EINSUMS_STREAM_DECLARE_RUNG_ENTRY(arch_baseline)
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_V2)
EINSUMS_STREAM_DECLARE_RUNG_ENTRY(arch_v2)
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_V3)
EINSUMS_STREAM_DECLARE_RUNG_ENTRY(arch_v3)
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_V4)
EINSUMS_STREAM_DECLARE_RUNG_ENTRY(arch_v4)
#endif

#undef EINSUMS_STREAM_DECLARE_RUNG_ENTRY

namespace {

// Ladder slot macros: nullptr keeps the position of rungs that were not built.
// arch_native (single-TU aarch64/pinned builds) occupies the baseline slot.
#if defined(EINSUMS_SIMD_HAS_RUNG_NATIVE)
#    define EINSUMS_STREAM_BASELINE_ENTRY(fn) &arch_native::fn
#elif defined(EINSUMS_SIMD_HAS_RUNG_BASELINE)
#    define EINSUMS_STREAM_BASELINE_ENTRY(fn) &arch_baseline::fn
#else
#    define EINSUMS_STREAM_BASELINE_ENTRY(fn) nullptr
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_V2)
#    define EINSUMS_STREAM_V2_ENTRY(fn) &arch_v2::fn
#else
#    define EINSUMS_STREAM_V2_ENTRY(fn) nullptr
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_V3)
#    define EINSUMS_STREAM_V3_ENTRY(fn) &arch_v3::fn
#else
#    define EINSUMS_STREAM_V3_ENTRY(fn) nullptr
#endif
#if defined(EINSUMS_SIMD_HAS_RUNG_V4)
#    define EINSUMS_STREAM_V4_ENTRY(fn) &arch_v4::fn
#else
#    define EINSUMS_STREAM_V4_ENTRY(fn) nullptr
#endif

#define EINSUMS_STREAM_LADDER(fn)                                                                                                          \
    EINSUMS_STREAM_BASELINE_ENTRY(fn), EINSUMS_STREAM_V2_ENTRY(fn), EINSUMS_STREAM_V3_ENTRY(fn), EINSUMS_STREAM_V4_ENTRY(fn)

} // namespace

#define EINSUMS_STREAM_DEFINE_ENTRY(T)                                                                                                     \
    template <>                                                                                                                            \
    EINSUMS_EXPORT StreamInnerFn<T> stream_inner_entry<T>() {                                                                              \
        static StreamInnerFn<T> const fn = einsums::simd::select<StreamInnerFn<T>>(EINSUMS_STREAM_LADDER(stream_inner<T>));                \
        return fn;                                                                                                                         \
    }

EINSUMS_STREAM_DEFINE_ENTRY(float)
EINSUMS_STREAM_DEFINE_ENTRY(double)
EINSUMS_STREAM_DEFINE_ENTRY(std::complex<float>)
EINSUMS_STREAM_DEFINE_ENTRY(std::complex<double>)

#undef EINSUMS_STREAM_DEFINE_ENTRY
#undef EINSUMS_STREAM_LADDER

} // namespace einsums::compute_graph::passes
