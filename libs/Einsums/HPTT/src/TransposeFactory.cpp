//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Arch-neutral plan factory: the one place the SIMD dispatch rung is chosen.
//
// The transpose implementation (Transpose.cpp) is compiled once per
// instruction-set rung by einsums_add_simd_dispatch_sources(), each copy in
// its own namespace (hptt::arch_baseline, hptt::arch_v3, ...). This TU is
// compiled exactly once, WITHOUT arch flags, and bridges the copies: it
// declares each rung's make_transpose entry points (guarded by the
// EINSUMS_SIMD_HAS_RUNG_* definitions the CMake helper emits) and picks the
// best one at or below einsums::simd::selected_arch(). Everything downstream
// of plan creation stays inside the chosen rung - execute() is a virtual
// call on the rung's TransposeImpl.

#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/HPTT/Transpose.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <cstdio>
#include <memory>

namespace hptt {

#define EINSUMS_HPTT_DECLARE_RUNG_FACTORIES(ns)                                                                                            \
    namespace ns {                                                                                                                         \
    template <typename floatType>                                                                                                          \
    std::shared_ptr<hptt::Transpose<floatType>>                                                                                            \
    make_transpose(size_t const *sizeA, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB, size_t const *offsetA,        \
                   size_t const *offsetB, size_t const innerStrideA, size_t const innerStrideB, int const dim, floatType const *A,         \
                   floatType const alpha, floatType *B, floatType const beta, SelectionMethod const selectionMethod, int const numThreads, \
                   int const *threadIds, bool const useRowMajor);                                                                          \
    template <typename floatType>                                                                                                          \
    std::shared_ptr<hptt::Transpose<floatType>> make_transpose_from_file(std::FILE *fp, floatType alpha, floatType const *A,               \
                                                                         floatType beta, floatType *B);                                    \
    }

#if defined(EINSUMS_SIMD_HAS_RUNG_NATIVE)
EINSUMS_HPTT_DECLARE_RUNG_FACTORIES(arch_native)
#else
#    if defined(EINSUMS_SIMD_HAS_RUNG_BASELINE)
EINSUMS_HPTT_DECLARE_RUNG_FACTORIES(arch_baseline)
#    endif
#    if defined(EINSUMS_SIMD_HAS_RUNG_V2)
EINSUMS_HPTT_DECLARE_RUNG_FACTORIES(arch_v2)
#    endif
#    if defined(EINSUMS_SIMD_HAS_RUNG_V3)
EINSUMS_HPTT_DECLARE_RUNG_FACTORIES(arch_v3)
#    endif
#    if defined(EINSUMS_SIMD_HAS_RUNG_V4)
EINSUMS_HPTT_DECLARE_RUNG_FACTORIES(arch_v4)
#    endif
#endif

#undef EINSUMS_HPTT_DECLARE_RUNG_FACTORIES

namespace {

template <typename floatType, typename Fn>
Fn select_rung(Fn native, Fn baseline, Fn v2, Fn v3, Fn v4) {
    if (native != nullptr) {
        return native;
    }
    return einsums::simd::select<Fn>(baseline, v2, v3, v4);
}

// Resolve the per-rung entry points for one element type. The nullptr slots
// keep the ladder positions of rungs that were not built.
#if defined(EINSUMS_SIMD_HAS_RUNG_NATIVE)
#    define EINSUMS_HPTT_RUNG_ENTRY(ns_fn) &arch_native::ns_fn
#    define EINSUMS_HPTT_LADDER(ns_fn)     EINSUMS_HPTT_RUNG_ENTRY(ns_fn), nullptr, nullptr, nullptr, nullptr
#else
#    if defined(EINSUMS_SIMD_HAS_RUNG_BASELINE)
#        define EINSUMS_HPTT_BASELINE_ENTRY(ns_fn) &arch_baseline::ns_fn
#    else
#        define EINSUMS_HPTT_BASELINE_ENTRY(ns_fn) nullptr
#    endif
#    if defined(EINSUMS_SIMD_HAS_RUNG_V2)
#        define EINSUMS_HPTT_V2_ENTRY(ns_fn) &arch_v2::ns_fn
#    else
#        define EINSUMS_HPTT_V2_ENTRY(ns_fn) nullptr
#    endif
#    if defined(EINSUMS_SIMD_HAS_RUNG_V3)
#        define EINSUMS_HPTT_V3_ENTRY(ns_fn) &arch_v3::ns_fn
#    else
#        define EINSUMS_HPTT_V3_ENTRY(ns_fn) nullptr
#    endif
#    if defined(EINSUMS_SIMD_HAS_RUNG_V4)
#        define EINSUMS_HPTT_V4_ENTRY(ns_fn) &arch_v4::ns_fn
#    else
#        define EINSUMS_HPTT_V4_ENTRY(ns_fn) nullptr
#    endif
#    define EINSUMS_HPTT_LADDER(ns_fn)                                                                                                     \
        nullptr, EINSUMS_HPTT_BASELINE_ENTRY(ns_fn), EINSUMS_HPTT_V2_ENTRY(ns_fn), EINSUMS_HPTT_V3_ENTRY(ns_fn),                           \
            EINSUMS_HPTT_V4_ENTRY(ns_fn)
#endif

} // namespace

template <typename floatType>
std::shared_ptr<Transpose<floatType>>
Transpose<floatType>::create(size_t const *sizeA, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB,
                             size_t const *offsetA, size_t const *offsetB, size_t const innerStrideA, size_t const innerStrideB,
                             int const dim, floatType const *A, floatType const alpha, floatType *B, floatType const beta,
                             SelectionMethod const selectionMethod, int const numThreads, int const *threadIds, bool const useRowMajor) {
    using Fn = std::shared_ptr<Transpose<floatType>> (*)(
        size_t const *, int const *, size_t const *, size_t const *, size_t const *, size_t const *, size_t const, size_t const, int const,
        floatType const *, floatType const, floatType *, floatType const, SelectionMethod const, int const, int const *, bool const);
    static Fn const fn = select_rung<floatType, Fn>(EINSUMS_HPTT_LADDER(make_transpose<floatType>));
    return fn(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB, dim, A, alpha, B, beta, selectionMethod,
              numThreads, threadIds, useRowMajor);
}

template <typename floatType>
std::shared_ptr<Transpose<floatType>> Transpose<floatType>::read_from_file(std::FILE *fp, floatType alpha, floatType const *A,
                                                                           floatType beta, floatType *B) {
    using Fn           = std::shared_ptr<Transpose<floatType>> (*)(std::FILE *, floatType, floatType const *, floatType, floatType *);
    static Fn const fn = select_rung<floatType, Fn>(EINSUMS_HPTT_LADDER(make_transpose_from_file<floatType>));
    return fn(fp, alpha, A, beta, B);
}

template class EINSUMS_EXPORT Transpose<float>;
template class EINSUMS_EXPORT Transpose<double>;
template class EINSUMS_EXPORT Transpose<FloatComplex>;
template class EINSUMS_EXPORT Transpose<DoubleComplex>;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
template class EINSUMS_EXPORT Transpose<einsums::simd::half_t>;
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
template class EINSUMS_EXPORT Transpose<einsums::simd::bfloat16_t>;
#endif

} // namespace hptt
