//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file symm_dispatch.cpp
/// @brief Tests for symmetry-aware gemm dispatch. Each test constructs
/// a symmetric/Hermitian input, runs gemm both with and without the symmetry
/// descriptor attached, and verifies the results agree to within tolerance.

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>

#include <Einsums/Testing.hpp>

using namespace einsums;
using linear_algebra::gemm;

TEMPLATE_TEST_CASE("symm dispatch - symmetric A matches general gemm", "[linear-algebra][symmetry]", float, double) {
    using T         = TestType;
    constexpr int N = 6, K = 5;

    auto A_sym = create_random_tensor<T>(true, "A", N, N);
    A_sym.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));
    symmetrize(A_sym); // Ensure the stored data matches the declared symmetry.

    auto B     = create_random_tensor<T>(true, "B", N, K);
    auto C_sym = create_zero_tensor<T>(true, "Csym", N, K);
    auto C_ref = create_zero_tensor<T>(true, "Cref", N, K);

    // Dispatched path: A carries the symmetric descriptor → symm is used.
    gemm('n', 'n', T{1}, A_sym, B, T{0}, &C_sym);

    // Reference path: clone A without the descriptor → general gemm.
    auto A_plain = create_zero_tensor<T>(true, "Aplain", N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_plain(i, j) = A_sym(i, j);
    gemm('n', 'n', T{1}, A_plain, B, T{0}, &C_ref);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j)
            REQUIRE(C_sym(i, j) == Catch::Approx(C_ref(i, j)).margin(::einsums::tolerance<T>()));
}

TEMPLATE_TEST_CASE("symm dispatch - symmetric B on the right matches general gemm", "[linear-algebra][symmetry]", float, double) {
    using T         = TestType;
    constexpr int M = 4, N = 5;

    auto A     = create_random_tensor<T>(true, "A", M, N);
    auto B_sym = create_random_tensor<T>(true, "B", N, N);
    B_sym.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));
    symmetrize(B_sym);

    auto C_sym = create_zero_tensor<T>(true, "Csym", M, N);
    auto C_ref = create_zero_tensor<T>(true, "Cref", M, N);

    gemm('n', 'n', T{1}, A, B_sym, T{0}, &C_sym);

    auto B_plain = create_zero_tensor<T>(true, "Bplain", N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            B_plain(i, j) = B_sym(i, j);
    gemm('n', 'n', T{1}, A, B_plain, T{0}, &C_ref);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            REQUIRE(C_sym(i, j) == Catch::Approx(C_ref(i, j)).margin(::einsums::tolerance<T>()));
}

TEMPLATE_TEST_CASE("symm dispatch - beta != 0 adds correctly", "[linear-algebra][symmetry]", float, double) {
    using T         = TestType;
    constexpr int N = 4;

    auto A_sym = create_random_tensor<T>(true, "A", N, N);
    A_sym.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));
    symmetrize(A_sym);

    auto B     = create_random_tensor<T>(true, "B", N, N);
    auto C_sym = create_random_tensor<T>(true, "Csym", N, N);
    auto C_ref = create_zero_tensor<T>(true, "Cref", N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C_ref(i, j) = C_sym(i, j);

    gemm('n', 'n', T{2}, A_sym, B, T{3}, &C_sym);

    auto A_plain = create_zero_tensor<T>(true, "Aplain", N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_plain(i, j) = A_sym(i, j);
    gemm('n', 'n', T{2}, A_plain, B, T{3}, &C_ref);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            REQUIRE(C_sym(i, j) == Catch::Approx(C_ref(i, j)).margin(::einsums::tolerance<T>()));
}

TEMPLATE_TEST_CASE("hemm dispatch - Hermitian A matches general gemm", "[linear-algebra][symmetry]", std::complex<float>,
                   std::complex<double>) {
    using T         = TestType;
    constexpr int N = 5, K = 4;

    auto A_her = create_random_tensor<T>(true, "A", N, N);
    A_her.set_symmetry(SymmetryDescriptor::hermitian_pair(0, 1));
    symmetrize(A_her);

    auto B     = create_random_tensor<T>(true, "B", N, K);
    auto C_her = create_zero_tensor<T>(true, "Cher", N, K);
    auto C_ref = create_zero_tensor<T>(true, "Cref", N, K);

    gemm('n', 'n', T{1}, A_her, B, T{0}, &C_her);

    auto A_plain = create_zero_tensor<T>(true, "Aplain", N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_plain(i, j) = A_her(i, j);
    gemm('n', 'n', T{1}, A_plain, B, T{0}, &C_ref);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            auto diff = C_her(i, j) - C_ref(i, j);
            REQUIRE(std::abs(diff) < ::einsums::tolerance<T>());
        }
    }
}

TEMPLATE_TEST_CASE("symm dispatch - trans='T' on symmetric A still dispatches", "[linear-algebra][symmetry]", float, double) {
    // For symmetric A, A^T = A, so gemm('T', 'N', alpha, A, B) == gemm('N', 'N', alpha, A, B).
    // The dispatch should recognize symmetric A and use symm regardless of the flag.
    using T         = TestType;
    constexpr int N = 5;

    auto A_sym = create_random_tensor<T>(true, "A", N, N);
    A_sym.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));
    symmetrize(A_sym);

    auto B       = create_random_tensor<T>(true, "B", N, N);
    auto C_trans = create_zero_tensor<T>(true, "Ctrans", N, N);
    auto C_ref   = create_zero_tensor<T>(true, "Cref", N, N);

    // trans='T' on A; should still dispatch to symm.
    gemm('t', 'n', T{1}, A_sym, B, T{0}, &C_trans);

    // Same call with explicit 'n' trans on a plain (non-symmetric-tagged) copy.
    auto A_plain = create_zero_tensor<T>(true, "Aplain", N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_plain(i, j) = A_sym(i, j);
    gemm('n', 'n', T{1}, A_plain, B, T{0}, &C_ref);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            REQUIRE(C_trans(i, j) == Catch::Approx(C_ref(i, j)).margin(::einsums::tolerance<T>()));
}

TEMPLATE_TEST_CASE("hemm dispatch - trans='C' on Hermitian A still dispatches", "[linear-algebra][symmetry]", std::complex<float>,
                   std::complex<double>) {
    // For Hermitian A, A^H = A, so gemm('C', 'N', alpha, A, B) == gemm('N', 'N', alpha, A, B).
    using T         = TestType;
    constexpr int N = 4;

    auto A_her = create_random_tensor<T>(true, "A", N, N);
    A_her.set_symmetry(SymmetryDescriptor::hermitian_pair(0, 1));
    symmetrize(A_her);

    auto B      = create_random_tensor<T>(true, "B", N, N);
    auto C_conj = create_zero_tensor<T>(true, "Cconj", N, N);
    auto C_ref  = create_zero_tensor<T>(true, "Cref", N, N);

    gemm('c', 'n', T{1}, A_her, B, T{0}, &C_conj);

    auto A_plain = create_zero_tensor<T>(true, "Aplain", N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_plain(i, j) = A_her(i, j);
    gemm('n', 'n', T{1}, A_plain, B, T{0}, &C_ref);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            REQUIRE(std::abs(C_conj(i, j) - C_ref(i, j)) < ::einsums::tolerance<T>());
}

TEMPLATE_TEST_CASE("symm dispatch - no descriptor falls through to gemm", "[linear-algebra][symmetry]", float, double) {
    using T         = TestType;
    constexpr int N = 3;

    auto A      = create_random_tensor<T>(true, "A", N, N);
    auto B      = create_random_tensor<T>(true, "B", N, N);
    auto C_auto = create_zero_tensor<T>(true, "Cauto", N, N);
    auto C_ref  = create_zero_tensor<T>(true, "Cref", N, N);

    // No descriptor set; should take the general path and compute correctly.
    gemm('n', 'n', T{1}, A, B, T{0}, &C_auto);
    gemm('n', 'n', T{1}, A, B, T{0}, &C_ref);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            REQUIRE(C_auto(i, j) == Catch::Approx(C_ref(i, j)).margin(::einsums::tolerance<T>()));
}

namespace {

/// ``op(A) op(B)`` by loops, reading each operand through its own flag: ``'n'``, ``'t'`` or ``'c'``.
template <typename T>
Tensor<T, 2> product_by_loops(char ta, Tensor<T, 2> const &A, char tb, Tensor<T, 2> const &B, bool row_major) {
    auto const op = [](char t, Tensor<T, 2> const &M, size_t i, size_t j) -> T {
        T value = (t == 'n') ? M(i, j) : M(j, i);
        if constexpr (IsComplexV<T>) {
            if (t == 'c') {
                value = std::conj(value);
            }
        }
        return value;
    };
    size_t const m = ta == 'n' ? A.dim(0) : A.dim(1);
    size_t const k = ta == 'n' ? A.dim(1) : A.dim(0);
    size_t const n = tb == 'n' ? B.dim(1) : B.dim(0);
    auto         C = create_zero_tensor<T>(row_major, "C_loops", m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum{0};
            for (size_t p = 0; p < k; ++p) {
                sum += op(ta, A, i, p) * op(tb, B, p, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

template <typename T>
void require_close_to_loops(Tensor<T, 2> const &got, Tensor<T, 2> const &want) {
    for (size_t i = 0; i < want.dim(0); ++i) {
        for (size_t j = 0; j < want.dim(1); ++j) {
            CAPTURE(i, j);
            REQUIRE(std::abs(got(i, j) - want(i, j)) < 100 * ::einsums::tolerance<T>());
        }
    }
}

} // namespace

TEMPLATE_TEST_CASE("symm dispatch - a complex symmetric operand under 'c' is its conjugate, not itself", "[linear-algebra][symmetry]",
                   std::complex<float>, std::complex<double>) {
    // A complex SYMMETRIC matrix has A^T = A but A^H = conj(A). The dispatch took 'c' on a
    // symmetric operand for 'n', as it may for a real one, and handed A itself to symm, so
    // gemm('c', 'n', A, B) computed A B instead of conj(A) B.
    using T         = TestType;
    constexpr int N = 4;
    for (bool const row_major : {false, true}) {
        CAPTURE(row_major);
        auto A = create_random_tensor<T>(row_major, "A", N, N);
        A.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));
        symmetrize(A);
        auto B = create_random_tensor<T>(row_major, "B", N, N);
        auto C = create_zero_tensor<T>(row_major, "C", N, N);

        gemm('c', 'n', T{1}, A, B, T{0}, &C);
        require_close_to_loops(C, product_by_loops('c', A, 'n', B, row_major));

        // The same operand on the right.
        auto D = create_zero_tensor<T>(row_major, "D", N, N);
        gemm('n', 'c', T{1}, B, A, T{0}, &D);
        require_close_to_loops(D, product_by_loops('n', B, 'c', A, row_major));
    }
}

TEMPLATE_TEST_CASE("symm dispatch - the other operand keeps its transpose", "[linear-algebra][symmetry]", float, double,
                   std::complex<float>, std::complex<double>) {
    // symm and hemm multiply the symmetric operand by the OTHER one as stored. The dispatch fired on
    // the symmetric operand's flag alone, so gemm('n', 't', A, B) with a symmetric A computed A B
    // rather than A B^T, and likewise with the symmetric operand on the right.
    using T         = TestType;
    constexpr int N = 4;
    for (bool const row_major : {false, true}) {
        CAPTURE(row_major);
        for (bool const hermitian : {false, true}) {
            if (hermitian && !IsComplexV<T>) {
                continue;
            }
            CAPTURE(hermitian);
            auto S = create_random_tensor<T>(row_major, "S", N, N);
            S.set_symmetry(hermitian ? SymmetryDescriptor::hermitian_pair(0, 1) : SymmetryDescriptor::symmetric_pair(0, 1));
            symmetrize(S);
            auto B = create_random_tensor<T>(row_major, "B", N, N);

            auto C = create_zero_tensor<T>(row_major, "C", N, N);
            gemm('n', 't', T{1}, S, B, T{0}, &C);
            require_close_to_loops(C, product_by_loops('n', S, 't', B, row_major));

            auto D = create_zero_tensor<T>(row_major, "D", N, N);
            gemm('t', 'n', T{1}, B, S, T{0}, &D);
            require_close_to_loops(D, product_by_loops('t', B, 'n', S, row_major));

            if constexpr (IsComplexV<T>) {
                auto E = create_zero_tensor<T>(row_major, "E", N, N);
                gemm('n', 'c', T{1}, S, B, T{0}, &E);
                require_close_to_loops(E, product_by_loops('n', S, 'c', B, row_major));
            }
        }
    }
}
