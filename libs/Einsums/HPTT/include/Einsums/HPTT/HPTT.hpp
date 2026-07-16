//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/**
 * @mainpage High-Performance Tensor Transpose (HPTT)
 *
 * HPTT supports tensor transpositions of the form:
 * \f[ B_{\pi(i_0,i_1,...)} = \alpha * A_{i_0,i_1,...} + \beta * B_{\pi(i_0,i_1,...)}. \f]
 *
 * @code
 *     auto plan = hptt::create_plan(perm, dim, alpha, A, size, nullptr,
 *                                   beta, B, nullptr, hptt::ESTIMATE, numThreads);
 *     plan->execute();
 * @endcode
 *
 * Supported types: float, double, std::complex<float>, std::complex<double>.
 *
 * @see hptt::Transpose, hptt::create_plan
 */

#pragma once

#include <Einsums/HPTT/Transpose.hpp>

#include <complex>
#include <memory>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace hptt {

// ===========================================================================
// Templatized create_plan overloads.
//
// These replace the former 36 (4 types × 9 signatures) non-template overloads
// with 9 function templates. Each body is a single make_shared call, so they
// are defined inline in the header.
// ===========================================================================

/// Create a transposition plan (basic).
template <typename T>
inline std::shared_ptr<Transpose<T>> create_plan(int const *perm, int dim, T alpha, T const *A, size_t const *sizeA,
                                                 size_t const *outerSizeA, T beta, T *B, size_t const *outerSizeB,
                                                 SelectionMethod selectionMethod, int numThreads, int const *threadIds = nullptr,
                                                 bool useRowMajor = false) {
    return Transpose<T>::create(sizeA, perm, outerSizeA, outerSizeB, nullptr, nullptr, 1, 1, dim, A, alpha, B, beta, selectionMethod,
                                numThreads, threadIds, useRowMajor);
}

/// Create a transposition plan (basic, vector arguments).
template <typename T>
inline std::shared_ptr<Transpose<T>> create_plan(std::vector<int> const &perm, int dim, T alpha, T const *A,
                                                 std::vector<size_t> const &sizeA, std::vector<size_t> const &outerSizeA, T beta, T *B,
                                                 std::vector<size_t> const &outerSizeB, SelectionMethod selectionMethod, int numThreads,
                                                 std::vector<int> const &threadIds = {}, bool useRowMajor = false) {
    return Transpose<T>::create(sizeA.data(), perm.data(), outerSizeA.data(), outerSizeB.data(), nullptr, nullptr, 1, 1, dim, A, alpha, B,
                                beta, selectionMethod, numThreads, threadIds.empty() ? nullptr : threadIds.data(), useRowMajor);
}

/// Create a transposition plan with offsets.
template <typename T>
inline std::shared_ptr<Transpose<T>> create_plan(int const *perm, int dim, T alpha, T const *A, size_t const *sizeA,
                                                 size_t const *outerSizeA, size_t const *offsetA, T beta, T *B, size_t const *outerSizeB,
                                                 size_t const *offsetB, SelectionMethod selectionMethod, int numThreads,
                                                 int const *threadIds = nullptr, bool useRowMajor = false) {
    return Transpose<T>::create(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, 1, 1, dim, A, alpha, B, beta, selectionMethod,
                                numThreads, threadIds, useRowMajor);
}

/// Create a transposition plan with offsets (vector arguments).
template <typename T>
inline std::shared_ptr<Transpose<T>> create_plan(std::vector<int> const &perm, int dim, T alpha, T const *A,
                                                 std::vector<size_t> const &sizeA, std::vector<size_t> const &outerSizeA,
                                                 std::vector<size_t> const &offsetA, T beta, T *B, std::vector<size_t> const &outerSizeB,
                                                 std::vector<size_t> const &offsetB, SelectionMethod selectionMethod, int numThreads,
                                                 std::vector<int> const &threadIds = {}, bool useRowMajor = false) {
    return Transpose<T>::create(sizeA.data(), perm.data(), outerSizeA.data(), outerSizeB.data(), offsetA.data(), offsetB.data(), 1, 1, dim,
                                A, alpha, B, beta, selectionMethod, numThreads, threadIds.empty() ? nullptr : threadIds.data(),
                                useRowMajor);
}

/// Create a transposition plan with offsets and inner strides.
template <typename T>
inline std::shared_ptr<Transpose<T>>
create_plan(int const *perm, int dim, T alpha, T const *A, size_t const *sizeA, size_t const *outerSizeA, size_t const *offsetA,
            size_t innerStrideA, T beta, T *B, size_t const *outerSizeB, size_t const *offsetB, size_t innerStrideB,
            SelectionMethod selectionMethod, int numThreads, int const *threadIds = nullptr, bool useRowMajor = false) {
    return Transpose<T>::create(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB, dim, A, alpha, B, beta,
                                selectionMethod, numThreads, threadIds, useRowMajor);
}

/// Create a transposition plan with offsets and inner strides (vector arguments).
template <typename T>
inline std::shared_ptr<Transpose<T>>
create_plan(std::vector<int> const &perm, int dim, T alpha, T const *A, std::vector<size_t> const &sizeA,
            std::vector<size_t> const &outerSizeA, std::vector<size_t> const &offsetA, size_t innerStrideA, T beta, T *B,
            std::vector<size_t> const &outerSizeB, std::vector<size_t> const &offsetB, size_t innerStrideB, SelectionMethod selectionMethod,
            int numThreads, std::vector<int> const &threadIds = {}, bool useRowMajor = false) {
    return Transpose<T>::create(sizeA.data(), perm.data(), outerSizeA.data(), outerSizeB.data(), offsetA.data(), offsetB.data(),
                                innerStrideA, innerStrideB, dim, A, alpha, B, beta, selectionMethod, numThreads,
                                threadIds.empty() ? nullptr : threadIds.data(), useRowMajor);
}

} // namespace hptt
