//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Arch-neutral plan factory: the one place the SIMD dispatch rung is chosen.
//
// The kernels are compiled per rung, each in its own namespace; the planner and this TU once. This
// picks the best built rung the machine supports and hands its kernel table to TransposeImpl.

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/HPTT/Transpose.hpp>

#include <Stripes/RungLadder.hpp>
#include <Stripes/RuntimeFeatures.hpp>
#include <cstdio>
#include <memory>

#include "TransposeImpl.hpp"
#include "TransposeKernels.hpp"

EINSUMS_NAMESPACE_BEGIN(hptt)

template <typename floatType>
TransposeKernels<floatType> const &selected_transpose_kernels() {
    using Fn                                    = TransposeKernels<floatType> const &(*)() noexcept;
    static TransposeKernels<floatType> const &k = stripes::select<Fn>(STRIPES_LADDER(transpose_kernels<floatType>))();
    return k;
}

template <typename floatType>
std::shared_ptr<Transpose<floatType>>
Transpose<floatType>::create(size_t const *sizeA, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB,
                             size_t const *offsetA, size_t const *offsetB, size_t const innerStrideA, size_t const innerStrideB,
                             int const dim, floatType const *A, floatType const alpha, floatType *B, floatType const beta,
                             SelectionMethod const selectionMethod, int const numThreads, int const *threadIds, bool const useRowMajor) {
    return std::make_shared<TransposeImpl<floatType>>(selected_transpose_kernels<floatType>(), sizeA, perm, outerSizeA, outerSizeB, offsetA,
                                                      offsetB, innerStrideA, innerStrideB, dim, A, alpha, B, beta, selectionMethod,
                                                      numThreads, threadIds, useRowMajor);
}

template <typename floatType>
std::shared_ptr<Transpose<floatType>> Transpose<floatType>::read_from_file(std::FILE *fp, floatType alpha, floatType const *A,
                                                                           floatType beta, floatType *B) {
    return std::make_shared<TransposeImpl<floatType>>(selected_transpose_kernels<floatType>(), fp, alpha, A, beta, B);
}

template class EINSUMS_EXPORT Transpose<float>;
template class EINSUMS_EXPORT Transpose<double>;
template class EINSUMS_EXPORT Transpose<FloatComplex>;
template class EINSUMS_EXPORT Transpose<DoubleComplex>;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
template class EINSUMS_EXPORT Transpose<stripes::half_t>;
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
template class EINSUMS_EXPORT Transpose<stripes::bfloat16_t>;
#endif

EINSUMS_NAMESPACE_END(hptt)
