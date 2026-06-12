//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Symmetry.cpp
/// @brief Phase 1 tests for SymmetryDescriptor + Tensor::{set_symmetry,
/// symmetrize,check_symmetry}. Phase 2 will add BLAS dispatch tests.

#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>

#include <Einsums/Testing.hpp>

using namespace einsums;

// ── SymmetryOp ────────────────────────────────────────────────────────────

TEST_CASE("SymmetryOp - identity", "[Tensor][Symmetry]") {
    auto op = SymmetryOp::identity();
    for (int i = 0; i < kMaxSymmetryRank; ++i)
        REQUIRE(op.permutation[i] == i);
    REQUIRE(op.sign == +1);
    REQUIRE_FALSE(op.conjugate);
}

TEST_CASE("SymmetryOp - swap", "[Tensor][Symmetry]") {
    auto op = SymmetryOp::swap(0, 1, -1);
    REQUIRE(op.permutation[0] == 1);
    REQUIRE(op.permutation[1] == 0);
    REQUIRE(op.permutation[2] == 2);
    REQUIRE(op.sign == -1);
}

TEST_CASE("SymmetryOp - group_swap for ERI bra-ket", "[Tensor][Symmetry]") {
    auto op = SymmetryOp::group_swap({0, 1}, {2, 3}, +1);
    REQUIRE(op.permutation[0] == 2);
    REQUIRE(op.permutation[1] == 3);
    REQUIRE(op.permutation[2] == 0);
    REQUIRE(op.permutation[3] == 1);
}

// ── SymmetryDescriptor factories ──────────────────────────────────────────

TEST_CASE("SymmetryDescriptor - factories produce expected generators", "[Tensor][Symmetry]") {
    auto s = SymmetryDescriptor::symmetric_pair(0, 1);
    REQUIRE(s.size() == 1);
    REQUIRE(s.ops[0].sign == +1);

    auto a = SymmetryDescriptor::antisymmetric_pair(0, 1);
    REQUIRE(a.size() == 1);
    REQUIRE(a.ops[0].sign == -1);

    auto h = SymmetryDescriptor::hermitian_pair(0, 1);
    REQUIRE(h.size() == 1);
    REQUIRE(h.ops[0].conjugate);

    auto eri = SymmetryDescriptor::eri_8fold();
    REQUIRE(eri.size() == 3); // inner-pair, inner-pair, bra-ket swap

    auto t2 = SymmetryDescriptor::ccsd_t2();
    REQUIRE(t2.size() == 2);
    REQUIRE(t2.ops[0].sign == -1);
    REQUIRE(t2.ops[1].sign == -1);
}

// ── Attach to Tensor ──────────────────────────────────────────────────────

TEST_CASE("Tensor - set_symmetry / clear_symmetry round trip", "[Tensor][Symmetry]") {
    auto A = create_zero_tensor<double>("A", 4, 4);
    REQUIRE_FALSE(A.has_symmetry());
    REQUIRE(A.symmetry() == nullptr);

    A.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));
    REQUIRE(A.has_symmetry());
    REQUIRE(A.symmetry()->size() == 1);

    A.clear_symmetry();
    REQUIRE_FALSE(A.has_symmetry());
    REQUIRE(A.symmetry() == nullptr);
}

TEST_CASE("Tensor - set_symmetry with empty descriptor clears", "[Tensor][Symmetry]") {
    auto A = create_zero_tensor<double>("A", 4, 4);
    A.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));
    REQUIRE(A.has_symmetry());
    A.set_symmetry(SymmetryDescriptor{}); // empty
    REQUIRE_FALSE(A.has_symmetry());
}

// ── symmetrize() ──────────────────────────────────────────────────────────

TEST_CASE("symmetrize - rank 2 symmetric average", "[Tensor][Symmetry]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    A.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));

    symmetrize(A);

    // After symmetrize, A(i,j) == A(j,i) exactly.
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            REQUIRE(A(i, j) == Catch::Approx(A(j, i)).margin(1e-14));
}

TEST_CASE("symmetrize - rank 2 antisymmetric zeros diagonal", "[Tensor][Symmetry]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    A.set_symmetry(SymmetryDescriptor::antisymmetric_pair(0, 1));

    symmetrize(A);

    // Diagonal of an antisymmetric tensor must be zero.
    for (int i = 0; i < 4; ++i)
        REQUIRE(A(i, i) == Catch::Approx(0.0).margin(1e-14));
    // Off-diagonal pairs are negatives.
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            REQUIRE(A(i, j) == Catch::Approx(-A(j, i)).margin(1e-14));
}

TEST_CASE("symmetrize - Hermitian complex tensor", "[Tensor][Symmetry]") {
    using cd = std::complex<double>;
    auto A   = create_random_tensor<cd>("A", 4, 4);
    A.set_symmetry(SymmetryDescriptor::hermitian_pair(0, 1));

    symmetrize(A);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            auto expected = std::conj(A(j, i));
            REQUIRE(std::abs(A(i, j) - expected) < 1e-14);
        }
    }
    // Diagonal of a Hermitian tensor has zero imaginary part.
    for (int i = 0; i < 4; ++i)
        REQUIRE(A(i, i).imag() == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE("symmetrize - rank 4 antisymmetric pair (CCSD T2 style)", "[Tensor][Symmetry]") {
    auto T = create_random_tensor<double>("T", 3, 3, 2, 2);
    // antisym in (0,1): T[a,b,i,j] = -T[b,a,i,j]
    T.set_symmetry(SymmetryDescriptor::antisymmetric_pair(0, 1));

    symmetrize(T);

    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    REQUIRE(T(a, b, i, j) == Catch::Approx(-T(b, a, i, j)).margin(1e-14));
}

TEST_CASE("symmetrize - rank 4 full CCSD T2 (antisym in both pairs)", "[Tensor][Symmetry]") {
    auto T = create_random_tensor<double>("T", 3, 3, 2, 2);
    T.set_symmetry(SymmetryDescriptor::ccsd_t2());

    symmetrize(T);

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    REQUIRE(T(a, b, i, j) == Catch::Approx(-T(b, a, i, j)).margin(1e-14));
                    REQUIRE(T(a, b, i, j) == Catch::Approx(-T(a, b, j, i)).margin(1e-14));
                    REQUIRE(T(a, b, i, j) == Catch::Approx(T(b, a, j, i)).margin(1e-14));
                }
            }
        }
    }
}

// ── check_symmetry() ──────────────────────────────────────────────────────

TEST_CASE("check_symmetry - passes after symmetrize", "[Tensor][Symmetry]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    A.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));

    REQUIRE_FALSE(check_symmetry(A)); // random tensor, not symmetric
    symmetrize(A);
    REQUIRE(check_symmetry(A));
}

TEST_CASE("check_symmetry - no descriptor returns true", "[Tensor][Symmetry]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    REQUIRE(check_symmetry(A));
}

TEST_CASE("check_symmetry - respects tolerance", "[Tensor][Symmetry]") {
    auto A = create_zero_tensor<double>("A", 3, 3);
    A.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));

    // Exact symmetry: trivially passes.
    REQUIRE(check_symmetry(A, 1e-14));

    // Introduce a small asymmetry.
    A(0, 1) = 1.0;
    A(1, 0) = 1.0 + 1e-10;

    REQUIRE_FALSE(check_symmetry(A, 1e-14));
    REQUIRE(check_symmetry(A, 1e-8));
}

TEST_CASE("check_symmetry - ERI 8-fold on a random rank-4 tensor", "[Tensor][Symmetry]") {
    auto E = create_random_tensor<double>("E", 3, 3, 3, 3);
    E.set_symmetry(SymmetryDescriptor::eri_8fold());

    // Random, definitely not ERI-symmetric.
    REQUIRE_FALSE(check_symmetry(E));

    symmetrize(E);
    REQUIRE(check_symmetry(E));

    // Spot-check each of the three generators' invariants.
    for (int m = 0; m < 3; ++m)
        for (int n = 0; n < 3; ++n)
            for (int l = 0; l < 3; ++l)
                for (int s = 0; s < 3; ++s) {
                    REQUIRE(E(m, n, l, s) == Catch::Approx(E(n, m, l, s)).margin(1e-12));
                    REQUIRE(E(m, n, l, s) == Catch::Approx(E(m, n, s, l)).margin(1e-12));
                    REQUIRE(E(m, n, l, s) == Catch::Approx(E(l, s, m, n)).margin(1e-12));
                }
}
