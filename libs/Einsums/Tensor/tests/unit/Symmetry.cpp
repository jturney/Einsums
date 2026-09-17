//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Symmetry.cpp
/// @brief Phase 1 tests for SymmetryDescriptor + Tensor::{set_symmetry,
/// symmetrize,check_symmetry}. Future work: BLAS dispatch tests.

#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>
#include <stdexcept>
#include <utility>
#include <vector>

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

// ── Runtime-rank forms ──────────────────────────────────────────────────────
//
// RuntimeTensor could already CARRY a descriptor through set_symmetry, with a
// comment beside it pointing at symmetrize / check_symmetry as the part that
// only covered statically ranked tensors. These cover the runtime-rank twins,
// which is the path the Python bindings and the ComputeGraph hold.

TEST_CASE("check_symmetry (runtime) - agrees with the statically ranked form", "[Tensor][Symmetry][Runtime]") {
    // The same data through both walks has to give the same verdict, or one of
    // them is wrong and there is no way to tell which.
    auto typed = create_random_tensor<double>("typed", 4, 4);
    auto rt    = RuntimeTensor<double>("rt", std::vector<size_t>{4, 4});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            rt(std::vector<size_t>{i, j}) = typed(i, j);
        }
    }

    auto const sym = SymmetryDescriptor::symmetric_pair(0, 1);
    typed.set_symmetry(sym);
    rt.set_symmetry(sym);

    REQUIRE(check_symmetry(typed) == check_symmetry(rt));
    REQUIRE_FALSE(check_symmetry(rt)); // random data is not symmetric

    symmetrize(typed);
    symmetrize(rt);
    REQUIRE(check_symmetry(typed));
    REQUIRE(check_symmetry(rt));

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            REQUIRE_THAT(rt(std::vector<size_t>{i, j}), Catch::Matchers::WithinAbs(typed(i, j), 1e-14));
        }
    }
}

TEST_CASE("check_symmetry (runtime) - a supplied descriptor, not the tensor's own", "[Tensor][Symmetry][Runtime]") {
    // The overload an optimizer needs: it is considering a symmetry and wants to
    // know whether it holds, which is a different question from whether the
    // tensor's own declared symmetry holds.
    auto rt = RuntimeTensor<double>("rt", std::vector<size_t>{3, 3, 3});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                // Antisymmetric in (0,1) by construction, nothing else. The
                // trailing factor has to MULTIPLY rather than add, or it
                // survives the swap and breaks the antisymmetry it decorates.
                double const v = (static_cast<double>(i) - static_cast<double>(j)) * (1.0 + 0.25 * static_cast<double>(k));
                rt(std::vector<size_t>{i, j, k}) = v;
            }
        }
    }
    REQUIRE(rt.symmetry() == nullptr);
    REQUIRE(check_symmetry(rt)); // no declared symmetry is vacuously satisfied

    CHECK(check_symmetry(rt, SymmetryDescriptor::antisymmetric_pair(0, 1)));
    CHECK_FALSE(check_symmetry(rt, SymmetryDescriptor::symmetric_pair(0, 1)));
    CHECK_FALSE(check_symmetry(rt, SymmetryDescriptor::antisymmetric_pair(0, 2)));
}

TEST_CASE("check_symmetry (runtime) - a generator across unequal extents cannot hold", "[Tensor][Symmetry][Runtime]") {
    // The statically ranked walk never asks, because its callers pass square
    // tensors. A runtime-rank tensor can be any shape, and permuting axes of
    // different lengths indexes out of range rather than merely failing.
    auto rt = RuntimeTensor<double>("rt", std::vector<size_t>{2, 5});
    rt.zero();
    CHECK_FALSE(check_symmetry(rt, SymmetryDescriptor::symmetric_pair(0, 1)));

    rt.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));
    CHECK_THROWS_AS(symmetrize(rt), std::invalid_argument);
}

TEST_CASE("check_symmetry (runtime) - an empty tensor satisfies any descriptor", "[Tensor][Symmetry][Runtime]") {
    auto rt = RuntimeTensor<double>("rt", std::vector<size_t>{0, 0});
    CHECK(check_symmetry(rt, SymmetryDescriptor::antisymmetric_pair(0, 1)));
}

TEST_CASE("symmetrize (runtime) - rank-4 CCSD T2 pattern", "[Tensor][Symmetry][Runtime]") {
    size_t const n  = 3;
    auto         rt = RuntimeTensor<double>("t2", std::vector<size_t>{n, n, n, n});
    double       v  = 0.0;
    for (size_t a = 0; a < n; ++a) {
        for (size_t b = 0; b < n; ++b) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    rt(std::vector<size_t>{a, b, i, j}) = (v += 1.0);
                }
            }
        }
    }
    rt.set_symmetry(SymmetryDescriptor::ccsd_t2());
    REQUIRE_FALSE(check_symmetry(rt));
    symmetrize(rt);
    REQUIRE(check_symmetry(rt));

    // Spot-check the invariant directly rather than trusting the verifier alone.
    for (size_t a = 0; a < n; ++a) {
        for (size_t b = 0; b < n; ++b) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    REQUIRE_THAT(rt(std::vector<size_t>{a, b, i, j}),
                                 Catch::Matchers::WithinAbs(-rt(std::vector<size_t>{b, a, i, j}), 1e-12));
                    REQUIRE_THAT(rt(std::vector<size_t>{a, b, i, j}),
                                 Catch::Matchers::WithinAbs(-rt(std::vector<size_t>{a, b, j, i}), 1e-12));
                }
            }
        }
    }
}

TEST_CASE("check_symmetry (runtime) - a failing generator stops at the first violation", "[Tensor][Symmetry][Runtime]") {
    // Not a timing test. The walk used to visit every element of the tensor even
    // after a violation, because the verdict was carried in a captured flag and
    // nothing told the iteration to stop; a generator that does not hold cost a
    // full sweep. Detection probes many candidate generators against tensors that
    // mostly do not carry them, so "cheap when false" is the property that makes
    // it affordable, and a counting visitor pins it without depending on a clock.
    std::vector<size_t> const dims{6, 6, 6};
    std::vector<size_t> const strides{36, 6, 1};

    std::size_t visited  = 0;
    bool const  complete = detail::for_each_symmetry_pair(dims, strides, SymmetryOp::swap(0, 1), [&](std::size_t, std::size_t, bool) {
        ++visited;
        return visited < 3; // "violation" on the third pair
    });

    CHECK_FALSE(complete);
    CHECK(visited == 3);

    // For contrast, a visitor that never stops sees every unordered pair plus
    // every fixed point: 6*6*6 elements, of which the swap(0,1) partner-equal set
    // is the 6*6 with idx[0] == idx[1], leaving (216 - 36) / 2 = 90 pairs.
    std::size_t all = 0;
    bool const  ran = detail::for_each_symmetry_pair(dims, strides, SymmetryOp::swap(0, 1), [&](std::size_t, std::size_t, bool) {
        ++all;
        return true;
    });
    CHECK(ran);
    CHECK(all == 90 + 36);
}

TEST_CASE("check_symmetry (runtime) - a view over an impl is walkable", "[Tensor][Symmetry][Runtime]") {
    // A ComputeGraph pass reaches a bound tensor as RuntimeTensorView over the
    // handle's impl, and that type carries no descriptor of its own. The
    // caller-supplied-descriptor overload is exactly the one an optimizer wants,
    // so requiring `.symmetry()` on it shut out its only real consumer.
    size_t const n     = 4;
    auto         owner = RuntimeTensor<double>("owner", std::vector<size_t>{n, n});
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            owner(std::vector<size_t>{i, j}) = static_cast<double>(i) * static_cast<double>(j);
        }
    }
    RuntimeTensorView<double> const view{owner.impl()};

    STATIC_REQUIRE(RuntimeRankWalkable<RuntimeTensorView<double>>);
    STATIC_REQUIRE_FALSE(RuntimeRankSymmetryTensor<RuntimeTensorView<double>>);

    CHECK(check_symmetry(view, SymmetryDescriptor::symmetric_pair(0, 1))); // i*j is symmetric
    CHECK_FALSE(check_symmetry(view, SymmetryDescriptor::antisymmetric_pair(0, 1)));
    // And the view agrees with its owner, which does carry a descriptor.
    CHECK(check_symmetry(view, SymmetryDescriptor::symmetric_pair(0, 1)) ==
          check_symmetry(owner, SymmetryDescriptor::symmetric_pair(0, 1)));
}
