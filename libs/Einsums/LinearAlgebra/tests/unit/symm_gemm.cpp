//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/TensorCompare.hpp>

#include <algorithm>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

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

namespace {

// The two-GEMM bodies each typed overload held before all of them reduced to the one rank-erased
// entry, transcribed call for call: the same GEMM entries, the same temporaries, the same
// conjugated copy of B. The entry lands on exactly these bits for every case but one, and the cases
// below compare bytes. The kernel choices that could move a bit are the ones in play: which gemm
// entry runs (the tensor-object one consults a declared symmetry), and the layout of the conjugated
// copy of B^T.
//
// That copy is where the one case differs. The entry lays the copy out like B. The transcribed body
// copies through RemoveViewT, which for a TensorView is the owning Tensor and lays the copy out
// densely in the default storage order, so for a TensorView B under the conjugating product with B^T the two
// hand the outer GEMM differently laid out operands and may round differently. That case is held to
// the brute-force congruence instead.

/// The compile-time-rank body.
template <bool TransA, bool TransB, typename AType, typename BType, typename CType>
void two_gemm_body(AType const &A, BType const &B, CType *C, bool conjugate) {
    using T             = typename AType::ValueType;
    int const temp_rows = static_cast<int>(TransA ? A.dim(1) : A.dim(0));
    int const temp_cols = static_cast<int>(TransB ? B.dim(0) : B.dim(1));
    *C                  = T(0.0);
    Tensor<T, 2> temp{"temp", temp_rows, temp_cols};
    linear_algebra::detail::gemm<TransA, TransB>(T{1.0}, A, B, T{0.0}, &temp);
    if (!conjugate) {
        linear_algebra::detail::gemm<!TransB, false>(T{1.0}, B, temp, T{0.0}, C);
    } else if constexpr (!TransB) {
        linear_algebra::detail::gemm('c', 'n', T{1.0}, B, temp, T{0.0}, C);
    } else {
        RemoveViewT<BType> Bc = B;
        einsums::detail::impl_conj(Bc.impl());
        linear_algebra::detail::gemm('n', 'n', T{1.0}, Bc, temp, T{0.0}, C);
    }
}

/// The runtime-rank body.
template <bool TransA, bool TransB, typename AType, typename BType, typename CType>
void two_gemm_body_runtime(AType const &A, BType const &B, CType *C, bool conjugate) {
    using T                = typename AType::ValueType;
    size_t const temp_rows = TransA ? A.dim(1) : A.dim(0);
    size_t const temp_cols = TransB ? B.dim(0) : B.dim(1);
    *C                     = T(0.0);
    Tensor<T, 2> temp{"temp", temp_rows, temp_cols};
    linear_algebra::gemm<TransA, TransB>(T{1.0}, A, B, T{0.0}, &temp);
    if (!conjugate) {
        linear_algebra::gemm<!TransB, false>(T{1.0}, B, temp, T{0.0}, C);
    } else if constexpr (!TransB) {
        linear_algebra::gemm('c', 'n', T{1.0}, B, temp, T{0.0}, C);
    } else {
        RemoveViewT<BType> Bc = B;
        einsums::detail::impl_conj(Bc.impl());
        linear_algebra::gemm('n', 'n', T{1.0}, Bc, temp, T{0.0}, C);
    }
}

template <bool TransA, bool TransB, typename AType, typename BType>
void require_compile_time_bits(AType const &A, BType const &B, size_t m, bool conjugate) {
    using T     = typename AType::ValueType;
    auto now    = create_zero_tensor<T>("C", m, m);
    auto before = create_zero_tensor<T>("C_before", m, m);
    if (conjugate) {
        linear_algebra::hermitian_symm_gemm<TransA, TransB>(A, B, &now);
    } else {
        linear_algebra::symm_gemm<TransA, TransB>(A, B, &now);
    }
    if (IsTensorViewV<BType> && TransB && conjugate) {
        // The one case whose copy is laid out differently; see the note above.
        Tensor<T, 2> const A_dense = A;
        Tensor<T, 2> const B_dense = B;
        size_t const       n       = B.dim(1);
        check_close(now, congruence<TransA, TransB>(A_dense, B_dense, n, m, conjugate && IsComplexV<T>));
        return;
    }
    two_gemm_body<TransA, TransB>(A, B, &before, conjugate);
    REQUIRE(testing::bytes_of(now) == testing::bytes_of(before));
}

template <bool TransA, bool TransB, typename AType, typename BType>
void require_runtime_bits(AType const &A, BType const &B, size_t m, bool conjugate) {
    using T = typename AType::ValueType;
    RuntimeTensor<T> now("C", std::vector<size_t>{m, m});
    RuntimeTensor<T> before("C_before", std::vector<size_t>{m, m});
    if (conjugate) {
        linear_algebra::hermitian_symm_gemm<TransA, TransB>(A, B, &now);
    } else {
        linear_algebra::symm_gemm<TransA, TransB>(A, B, &now);
    }
    two_gemm_body_runtime<TransA, TransB>(A, B, &before, conjugate);
    REQUIRE(testing::bytes_of(now) == testing::bytes_of(before));
}

/// Every operand kind the copy and the symmetry switch tell apart, for one transpose pair.
template <bool TransA, bool TransB, typename T>
void require_unchanged_bits() {
    constexpr size_t n  = 4;
    constexpr size_t m  = 3;
    size_t const     b0 = TransB ? m : n;
    size_t const     b1 = TransB ? n : m;

    auto const A         = create_random_tensor<T>("A", n, n);
    auto const B_col     = create_random_tensor<T>("B", b0, b1);
    auto const B_row     = create_random_tensor<T>(true, "B", b0, b1);
    auto       B_owner   = create_random_tensor<T>("B_owner", b1, b0);
    auto const B_view    = B_owner.transpose_view(); // row-major contiguous, a VIEW
    auto       B_declare = create_random_tensor<T>("B_sym", n, n);
    B_declare.set_symmetry(IsComplexV<T> ? SymmetryDescriptor::hermitian_pair(0, 1) : SymmetryDescriptor::symmetric_pair(0, 1));
    symmetrize(B_declare);

    RuntimeTensor<T> const rA(A);
    RuntimeTensor<T> const rB_col(B_col);
    RuntimeTensor<T> const rB_row(B_row);
    RuntimeTensor<T>       rB_owner(B_owner);
    auto const             rB_view = rB_owner.transpose_view();

    for (bool const conjugate : {false, true}) {
        CAPTURE(conjugate);
        INFO("compile-time rank: column-major, row-major, a view, a declared symmetry");
        require_compile_time_bits<TransA, TransB>(A, B_col, m, conjugate);
        require_compile_time_bits<TransA, TransB>(A, B_row, m, conjugate);
        require_compile_time_bits<TransA, TransB>(A, B_view, m, conjugate);
        require_compile_time_bits<TransA, TransB>(A, B_declare, n, conjugate);
        INFO("runtime rank: column-major, row-major, a view");
        require_runtime_bits<TransA, TransB>(rA, rB_col, m, conjugate);
        require_runtime_bits<TransA, TransB>(rA, rB_row, m, conjugate);
        require_runtime_bits<TransA, TransB>(rA, rB_view, m, conjugate);
    }
}

} // namespace

TEMPLATE_LIST_TEST_CASE("symm_gemm: the rank-erased entry computes the bits the typed overloads computed", "[linear-algebra][symm_gemm]",
                        testing::AllScalarTypes) {
    using T = TestType;
    SECTION("A, B") {
        require_unchanged_bits<false, false, T>();
    }
    SECTION("A^T, B") {
        require_unchanged_bits<true, false, T>();
    }
    SECTION("A, B^T") {
        require_unchanged_bits<false, true, T>();
    }
    SECTION("A^T, B^T") {
        require_unchanged_bits<true, true, T>();
    }
}

TEST_CASE("symm_gemm: the rank-erased entry refuses a transpose flag it cannot honour", "[linear-algebra][symm_gemm]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);
    CHECK_THROWS_AS(linear_algebra::detail::symm_gemm('c', 'n', false, A.impl(), B.impl(), &C.impl()), std::invalid_argument);
    CHECK_THROWS_AS(linear_algebra::detail::symm_gemm('n', 'x', true, A.impl(), B.impl(), &C.impl()), std::invalid_argument);
}

TEMPLATE_LIST_TEST_CASE("symm_gemm: the conjugating product over B^T leaves B as it found it", "[linear-algebra][symm_gemm]",
                        testing::ComplexScalarTypes) {
    // op(B)^H with op(B) = B^T is conj(B), formed in a conjugated copy of B. A RuntimeTensorView
    // names no underlying_type, so RemoveViewT of one is the view type itself, and copying that
    // object shares B's storage: conjugating the "copy" conjugated the caller's B in place. The
    // product was right and B came back conjugated, so a second identical call computed with a
    // different B.
    using T      = TestType;
    auto const A = create_random_tensor<T>("A", 4, 4);

    RuntimeTensor<T> rA(A);
    RuntimeTensor<T> owner(create_random_tensor<T>("owner", 4, 3));
    auto             view   = owner.transpose_view(); // B^T is 4 x 3, so B is the 3 x 4 view
    auto const       before = testing::bytes_of(owner);

    RuntimeTensor<T> C1("C1", std::vector<size_t>{3, 3});
    RuntimeTensor<T> C2("C2", std::vector<size_t>{3, 3});
    linear_algebra::hermitian_symm_gemm<false, true>(rA, view, &C1);
    REQUIRE(testing::bytes_of(owner) == before);
    linear_algebra::hermitian_symm_gemm<false, true>(rA, view, &C2);
    REQUIRE(testing::bytes_of(C1) == testing::bytes_of(C2));

    Tensor<T, 2> static_owner  = create_random_tensor<T>("static_owner", 4, 3);
    auto const   static_view   = static_owner.transpose_view();
    auto const   static_before = testing::bytes_of(static_owner);
    auto         C3            = create_zero_tensor<T>("C3", 3, 3);
    linear_algebra::hermitian_symm_gemm<false, true>(A, static_view, &C3);
    REQUIRE(testing::bytes_of(static_owner) == static_before);
}
