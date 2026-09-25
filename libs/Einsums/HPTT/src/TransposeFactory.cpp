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

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/HPTT/Transpose.hpp>
#include <Einsums/SIMD/RungLadder.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <cstdio>
#include <memory>

EINSUMS_NAMESPACE_BEGIN(hptt)

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

EINSUMS_SIMD_FOR_EACH_BUILT_RUNG(EINSUMS_HPTT_DECLARE_RUNG_FACTORIES)

#undef EINSUMS_HPTT_DECLARE_RUNG_FACTORIES

template <typename floatType>
std::shared_ptr<Transpose<floatType>>
Transpose<floatType>::create(size_t const *sizeA, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB,
                             size_t const *offsetA, size_t const *offsetB, size_t const innerStrideA, size_t const innerStrideB,
                             int const dim, floatType const *A, floatType const alpha, floatType *B, floatType const beta,
                             SelectionMethod const selectionMethod, int const numThreads, int const *threadIds, bool const useRowMajor) {
    using Fn = std::shared_ptr<Transpose<floatType>> (*)(
        size_t const *, int const *, size_t const *, size_t const *, size_t const *, size_t const *, size_t const, size_t const, int const,
        floatType const *, floatType const, floatType *, floatType const, SelectionMethod const, int const, int const *, bool const);
    static Fn const fn = einsums::simd::select<Fn>(EINSUMS_SIMD_LADDER(make_transpose<floatType>));
    return fn(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB, dim, A, alpha, B, beta, selectionMethod,
              numThreads, threadIds, useRowMajor);
}

template <typename floatType>
std::shared_ptr<Transpose<floatType>> Transpose<floatType>::read_from_file(std::FILE *fp, floatType alpha, floatType const *A,
                                                                           floatType beta, floatType *B) {
    using Fn           = std::shared_ptr<Transpose<floatType>> (*)(std::FILE *, floatType, floatType const *, floatType, floatType *);
    static Fn const fn = einsums::simd::select<Fn>(EINSUMS_SIMD_LADDER(make_transpose_from_file<floatType>));
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

EINSUMS_NAMESPACE_END(hptt)
