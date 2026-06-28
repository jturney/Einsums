//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("outer product", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    size_t _x{100}, _y{100};

    SECTION("1 * 1 -> 2") {
        Tensor A = create_random_tensor<TestType>("A", _x);
        Tensor B = create_random_tensor<TestType>("B", _y);
        Tensor C = create_zero_tensor<TestType>("C", _x, _y);
        zero(C);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i}, A, Indices{j}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                REQUIRE_THAT(C(x, y), CheckWithinRel(A(x) * B(y), 0.01));
                // REQUIRE_THAT(C(x, y), Catch::Matchers::WithinAbs(A(x) * B(y), 0.001));
            }
        }

        // C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j}, A, Indices{i}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                REQUIRE_THAT(C(x, y), CheckWithinRel(A(y) * B(x), 0.01));
                // REQUIRE_THAT(C(x, y), Catch::Matchers::WithinAbs(A(y) * B(x), 0.001));
            }
        }

        // C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{j}, A, Indices{i}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                REQUIRE_THAT(C(y, x), CheckWithinRel(A(y) * B(x), 0.01));
                // REQUIRE_THAT(C(y, x), Catch::Matchers::WithinAbs(A(y) * B(x), 0.001));
            }
        }

        // C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{i}, A, Indices{j}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                REQUIRE_THAT(C(y, x), CheckWithinRel(A(x) * B(y), 0.01));
                // REQUIRE_THAT(C(y, x), Catch::Matchers::WithinAbs(A(x) * B(y), 0.001));
            }
        }
    }

    SECTION("2 * 1 -> 3") {
        Tensor A = create_random_tensor<TestType>("A", 3, 3);
        Tensor B = create_random_tensor<TestType>("B", 3);
        Tensor C = create_tensor<TestType>("C", 3, 3, 3);

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k}, &C, Indices{i, j}, A, Indices{k}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    REQUIRE_THAT(C(x, y, z), CheckWithinRel(A(x, y) * B(z), 0.001));
                    // REQUIRE_THAT(C(x, y, z), Catch::Matchers::WithinAbs(A(x, y) * B(z), 0.001));
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{k, i, j}, &C, Indices{i, j}, A, Indices{k}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    REQUIRE_THAT(C(z, x, y), CheckWithinRel(A(x, y) * B(z), 0.001));
                    // REQUIRE_THAT(C(z, x, y), Catch::Matchers::WithinAbs(A(x, y) * B(z), 0.001));
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{k, i, j}, &C, Indices{k}, B, Indices{i, j}, A));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    REQUIRE_THAT(C(z, x, y), CheckWithinRel(A(x, y) * B(z), 0.001));
                    // REQUIRE_THAT(C(z, x, y), Catch::Matchers::WithinAbs(A(x, y) * B(z), 0.001));
                }
            }
        }
    }

    SECTION("2 * 2 -> 4") {
        Tensor A = create_random_tensor<TestType>("A", 3, 3);
        Tensor B = create_random_tensor<TestType>("B", 3, 3);
        Tensor C = create_tensor<TestType>("C", 3, 3, 3, 3);
        ;

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{i, j}, A, Indices{k, l}, B));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT(C(w, x, y, z), CheckWithinRel(A(w, x) * B(y, z), 0.001));
                        // REQUIRE_THAT(C(w, x, y, z), Catch::Matchers::WithinAbs(A(w, x) * B(y, z), 0.001));
                    }
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{k, l}, A, Indices{i, j}, B));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT(C(w, x, y, z), CheckWithinRel(A(y, z) * B(w, x), 0.001));
                        // REQUIRE_THAT(C(w, x, y, z), Catch::Matchers::WithinAbs(A(y, z) * B(w, x), 0.001));
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("view outer product", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    SECTION("1 * 1 -> 2") {
        Tensor A = create_random_tensor<TestType>("A", 6);
        Tensor B = create_random_tensor<TestType>("B", 6);

        auto   vA = TensorView(A, Dim{3}, Offset{3});
        auto   vB = TensorView(B, Dim{3});
        Tensor C  = create_zero_tensor<TestType>("C", 3, 3);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i}, vA, Indices{j}, vB));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT(C(x, y), CheckWithinRel(vA(x) * vB(y), 0.001));
                // REQUIRE_THAT(C(x, y), Catch::Matchers::WithinAbs(vA(x) * vB(y), 0.001));
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j}, vA, Indices{i}, vB));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT(C(x, y), CheckWithinRel(vA(y) * vB(x), 0.001));
                // REQUIRE_THAT(C(x, y), Catch::Matchers::WithinAbs(vA(y) * vB(x), 0.001));
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{j}, vA, Indices{i}, vB));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT(C(y, x), CheckWithinRel(vA(y) * vB(x), 0.001));
                // REQUIRE_THAT(C(y, x), Catch::Matchers::WithinAbs(vA(y) * vB(x), 0.001));
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{i}, vA, Indices{j}, vB));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT(C(y, x), CheckWithinRel(vA(x) * vB(y), 0.001));
                // REQUIRE_THAT(C(y, x), Catch::Matchers::WithinAbs(vA(x) * vB(y), 0.001));
            }
        }
    }

    SECTION("2 * 2 -> 4") {
        Tensor A  = create_random_tensor<TestType>("A", 9, 9);
        Tensor B  = create_random_tensor<TestType>("B", 12, 12);
        auto   vA = TensorView{A, Dim{3, 3}, Offset{6, 3}};
        auto   vB = TensorView{B, Dim{3, 3}, Offset{5, 7}};
        Tensor C  = create_zero_tensor<TestType>("C", 3, 3, 3, 3);

        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{i, j}, vA, Indices{k, l}, vB));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT(C(w, x, y, z), CheckWithinRel(vA(w, x) * vB(y, z), 0.001));
                        // REQUIRE_THAT(C(w, x, y, z), Catch::Matchers::WithinAbs(vA(w, x) * vB(y, z), 0.001));
                    }
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{k, l}, vA, Indices{i, j}, vB));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT(C(w, x, y, z), CheckWithinRel(vA(y, z) * vB(w, x), 0.001));
                        // REQUIRE_THAT(C(w, x, y, z), Catch::Matchers::WithinAbs(vA(y, z) * vB(w, x), 0.001));
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Issue #283: outer products (no contracted index) where a rank-2+ operand's
// indices are NON-CONTIGUOUS in the output. Every case above keeps each
// operand's indices contiguous in C ('ij,k->ijk', 'ij,kl->ijkl'); the eager
// outer-product path (einsum_do_outer_product) flattens C into a rank-2
// GER-style view assuming that contiguity, so interleaved targets produce a
// wrong result (dense) or run off the output's stride array (tiled).
//
// This is a sweep over the distinct non-contiguous rank-3/rank-4 orderings, so
// it doubles as a fix-completeness check: each ordering is its OWN test case,
// tagged [!shouldfail], so a partial fix (some orderings fixed, others not)
// shows up per ordering -- a fixed one starts passing and Catch2 turns the run
// RED, a reminder to drop that tag. The contiguous controls confirm the
// reference math. All non-contiguous cases below currently fail cleanly with
// wrong values (verified); the tiled variant instead aborts and is skipped.
// See https://github.com/Einsums/Einsums/pull/257.
// ----------------------------------------------------------------------------
namespace {
template <typename F>
void verify_outer3(einsums::Tensor<double, 3> const &C, F ref) {
    using namespace einsums; // CheckWithinRel
    for (int x = 0; x < 3; x++)
        for (int y = 0; y < 3; y++)
            for (int z = 0; z < 3; z++)
                REQUIRE_THAT(C(x, y, z), CheckWithinRel(ref(x, y, z), 0.001));
}
template <typename F>
void verify_outer4(einsums::Tensor<double, 4> const &C, F ref) {
    using namespace einsums; // CheckWithinRel
    for (int w = 0; w < 3; w++)
        for (int x = 0; x < 3; x++)
            for (int y = 0; y < 3; y++)
                for (int z = 0; z < 3; z++)
                    REQUIRE_THAT(C(w, x, y, z), CheckWithinRel(ref(w, x, y, z), 0.001));
}
} // namespace

TEST_CASE("outer product, contiguous controls (#283 sweep)", "[tensor_algebra][outer_product]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto v = create_random_tensor<double>("v", 3);

    SECTION("ab,c->abc") {
        auto C = create_tensor<double>("C", 3, 3, 3);
        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{a, b, c}, &C, Indices{a, b}, A, Indices{c}, v));
        verify_outer3(C, [&](int x, int y, int z) { return A(x, y) * v(z); });
    }
    SECTION("ab,cd->abcd") {
        auto C = create_tensor<double>("C", 3, 3, 3, 3);
        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{a, b, c, d}, &C, Indices{a, b}, A, Indices{c, d}, B));
        verify_outer4(C, [&](int w, int x, int y, int z) { return A(w, x) * B(y, z); });
    }
}

// Each non-contiguous ordering is its own [!shouldfail] case for per-ordering
// fix tracking. 'ac,b->abc' / 'b,ac->abc' straddle in rank 3; 'ac,bd->abcd'
// fully interleaves; 'ad,bc->abcd' straddles the outer operand around the
// inner; 'bc,ad->abcd' is the operand-swapped mirror.
TEST_CASE("outer product 'ac,b->abc' non-contiguous (#283)", "[tensor_algebra][outer_product][!shouldfail]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;
    auto A = create_random_tensor<double>("A", 3, 3);
    auto v = create_random_tensor<double>("v", 3);
    auto C = create_tensor<double>("C", 3, 3, 3);
    C.set_all(0.0);
    REQUIRE_NOTHROW(einsum(Indices{a, b, c}, &C, Indices{a, c}, A, Indices{b}, v));
    verify_outer3(C, [&](int x, int y, int z) { return A(x, z) * v(y); });
}
TEST_CASE("outer product 'b,ac->abc' non-contiguous (#283)", "[tensor_algebra][outer_product][!shouldfail]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;
    auto A = create_random_tensor<double>("A", 3, 3);
    auto v = create_random_tensor<double>("v", 3);
    auto C = create_tensor<double>("C", 3, 3, 3);
    C.set_all(0.0);
    REQUIRE_NOTHROW(einsum(Indices{a, b, c}, &C, Indices{b}, v, Indices{a, c}, A));
    verify_outer3(C, [&](int x, int y, int z) { return v(y) * A(x, z); });
}
TEST_CASE("outer product 'ac,bd->abcd' non-contiguous (#283)", "[tensor_algebra][outer_product][!shouldfail]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_tensor<double>("C", 3, 3, 3, 3);
    C.set_all(0.0);
    REQUIRE_NOTHROW(einsum(Indices{a, b, c, d}, &C, Indices{a, c}, A, Indices{b, d}, B));
    verify_outer4(C, [&](int w, int x, int y, int z) { return A(w, y) * B(x, z); });
}
TEST_CASE("outer product 'ad,bc->abcd' non-contiguous (#283)", "[tensor_algebra][outer_product][!shouldfail]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_tensor<double>("C", 3, 3, 3, 3);
    C.set_all(0.0);
    REQUIRE_NOTHROW(einsum(Indices{a, b, c, d}, &C, Indices{a, d}, A, Indices{b, c}, B));
    verify_outer4(C, [&](int w, int x, int y, int z) { return A(w, z) * B(x, y); });
}
TEST_CASE("outer product 'bc,ad->abcd' non-contiguous (#283)", "[tensor_algebra][outer_product][!shouldfail]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_tensor<double>("C", 3, 3, 3, 3);
    C.set_all(0.0);
    REQUIRE_NOTHROW(einsum(Indices{a, b, c, d}, &C, Indices{b, c}, A, Indices{a, d}, B));
    verify_outer4(C, [&](int w, int x, int y, int z) { return A(x, y) * B(w, z); });
}

TEST_CASE("tiled outer product, no contracted index (#283)", "[tensor_algebra][outer_product]") {
    // 'ia,jb->ijab' into a single-tile rank-4 TiledTensor (pure outer product,
    // operands interleaved in the output). On 1.1.2 this SIGSEGV'd; today it
    // throws std::out_of_range from inside the einsum and std::terminate()s --
    // the throw escapes through a context (an OpenMP region) that Catch2 cannot
    // intercept, so a [!shouldfail] guard does not help: the SIGABRT would take
    // down the whole test binary. Skip until the outer-product path handles
    // non-contiguous tiled targets (PR #257), then replace the SKIP with:
    //
    //   std::vector<int> occ{5}, vir{2};
    //   einsums::TiledTensor<double, 2> t1("t1", occ, vir);
    //   einsums::TiledTensor<double, 4> tau("tau", occ, occ, vir, vir);
    //   t1.tile(0, 0).zero(); tau.tile(0, 0, 0, 0).zero();
    //   using namespace einsums::index;
    //   REQUIRE_NOTHROW(einsum(einsums::tensor_algebra::Indices{i, j, a, b}, &tau,
    //                          einsums::tensor_algebra::Indices{i, a}, t1,
    //                          einsums::tensor_algebra::Indices{j, b}, t1));
    SKIP("#283: tiled outer product 'ia,jb->ijab' aborts the process (uncatchable); re-enable after the fix");
}
