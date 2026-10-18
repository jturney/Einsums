//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <algorithm>
#include <complex>
#include <limits>

#include <Einsums/Testing.hpp>

using namespace einsums;

namespace {

/// op(B)^T op(A) op(B), or op(B)^H op(A) op(B) when @p conjugate, by loops: op(A) is n x n and op(B)
/// is n x m, so C is m x m.
template <bool TransA, bool TransB, typename T>
Tensor<T, 2> congruence(Tensor<T, 2> const &A, Tensor<T, 2> const &B, size_t n, size_t m, bool conjugate) {
    auto const op_a = [&](size_t i, size_t j) { return TransA ? A(j, i) : A(i, j); };
    auto const op_b = [&](size_t i, size_t j) { return TransB ? B(j, i) : B(i, j); };
    auto       C    = create_zero_tensor<T>("C_ref", m, m);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            T sum{0};
            for (size_t k = 0; k < n; k++) {
                T outer = op_b(k, i);
                if constexpr (IsComplexV<T>) {
                    outer = conjugate ? std::conj(outer) : outer;
                }
                for (size_t l = 0; l < n; l++) {
                    sum += outer * op_a(k, l) * op_b(l, j);
                }
            }
            C(i, j) = sum;
        }
    }
    return C;
}

template <typename T>
void check_close(Tensor<T, 2> const &got, Tensor<T, 2> const &want) {
    double const tol = std::max(1e-12, 1000.0 * static_cast<double>(std::numeric_limits<RemoveComplexT<T>>::epsilon()));
    for (size_t i = 0; i < want.dim(0); i++) {
        for (size_t j = 0; j < want.dim(1); j++) {
            CAPTURE(i, j);
            REQUIRE(std::abs(got(i, j) - want(i, j)) <= tol * (1.0 + std::abs(want(i, j))));
        }
    }
}

/// Both functions, through the compile-time-rank overload and the runtime-rank one, with a B that is
/// not square: op(A) is 4 x 4 and op(B) is 4 x 3, so C is 3 x 3.
template <bool TransA, bool TransB, typename T>
void check_combination() {
    constexpr size_t n = 4;
    constexpr size_t m = 3;
    auto const       A = create_random_tensor<T>("A", n, n);
    auto const       B = TransB ? create_random_tensor<T>("B", m, n) : create_random_tensor<T>("B", n, m);

    for (bool const conjugate : {false, true}) {
        if (conjugate && !IsComplexV<T>) {
            continue;
        }
        CAPTURE(conjugate);
        auto const want = congruence<TransA, TransB>(A, B, n, m, conjugate);

        auto fixed = create_zero_tensor<T>("C", m, m);
        if (conjugate) {
            linear_algebra::hermitian_symm_gemm<TransA, TransB>(A, B, &fixed);
        } else {
            linear_algebra::symm_gemm<TransA, TransB>(A, B, &fixed);
        }
        check_close(fixed, want);

        RuntimeTensor<T> const rA{A};
        RuntimeTensor<T> const rB{B};
        RuntimeTensor<T>       rC{"C", {m, m}};
        if (conjugate) {
            linear_algebra::hermitian_symm_gemm<TransA, TransB>(rA, rB, &rC);
        } else {
            linear_algebra::symm_gemm<TransA, TransB>(rA, rB, &rC);
        }
        Tensor<T, 2> runtime = create_zero_tensor<T>("C_runtime", m, m);
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < m; j++) {
                runtime(i, j) = rC(i, j);
            }
        }
        check_close(runtime, want);
    }
}

} // namespace

TEMPLATE_LIST_TEST_CASE("symm_gemm: op(B)^T op(A) op(B) with a B that is not square", "[linear-algebra][symm_gemm]",
                        testing::AllScalarTypes) {
    // The compile-time-rank overload checked the shapes of op(B) op(A) op(B)^T, the product the other
    // way round, so it refused every B that was not square though the two GEMMs it then ran were the
    // right ones; the runtime-rank overload, which leaves the check to those GEMMs, was correct.
    using T = TestType;
    SECTION("A, B") {
        check_combination<false, false, T>();
    }
    SECTION("A^T, B") {
        check_combination<true, false, T>();
    }
    SECTION("A, B^T") {
        check_combination<false, true, T>();
    }
    SECTION("A^T, B^T") {
        check_combination<true, true, T>();
    }
}
