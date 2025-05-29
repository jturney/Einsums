//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("einsum3", "[tensor_algebra]", float, double) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    SECTION("3x3 <- 3x5 * 5x3") {
        auto C0 = create_tensor<TestType>("C0", 3, 3);
        auto C1 = create_tensor<TestType>("C1", 3, 3);
        auto A  = create_random_tensor<TestType>("A", 3, 5);
        auto B  = create_random_tensor<TestType>("B", 5, 3);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), TestType{0.0001}));
            }
        }
    }

    SECTION("3x3 <- 3x5 * 3x5") {
        auto C0 = create_tensor<TestType>("C0", 3, 3);
        auto C1 = create_tensor<TestType>("C1", 3, 3);
        auto A  = create_random_tensor<TestType>("A", 3, 5);
        auto B  = create_random_tensor<TestType>("B", 3, 5);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{j, k}, B));
        linear_algebra::gemm<false, true>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), TestType{0.0001}));
            }
        }
    }

    SECTION("3 <- 3x5 * 5") {
        auto C0 = create_tensor<TestType>("C0", 3);
        auto C1 = create_tensor<TestType>("C1", 3);
        auto A  = create_random_tensor<TestType>("A", 3, 5);
        auto B  = create_random_tensor<TestType>("B", 5);

        C0.zero();
        C1.zero();

        REQUIRE_NOTHROW(einsum(Indices{i}, &C0, Indices{i, j}, A, Indices{j}, B));
        linear_algebra::gemv<false>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            REQUIRE_THAT(C0(i0), Catch::Matchers::WithinAbs(C1(i0), TestType{0.001}));
        }
    }

    SECTION("3 <- 3x4x5 * 4x3x5") {
        auto C0 = create_tensor<TestType>("C0", 3);
        // zero(C0);
        auto C1 = create_tensor<TestType>("C1", 3);
        // zero(C1);
        auto A = create_random_tensor<TestType>("A", 3, 4, 5);
        auto B = create_random_tensor<TestType>("B", 4, 3, 5);

        REQUIRE_NOTHROW(einsum(Indices{i}, &C0, Indices{i, j, k}, A, Indices{j, i, k}, B));

        for (size_t i0 = 0; i0 < 3; i0++) {
            TestType sum{0};

            for (size_t j0 = 0; j0 < 4; j0++) {
                for (size_t k0 = 0; k0 < 5; k0++) {
                    sum += A(i0, j0, k0) * B(j0, i0, k0);
                }
            }
            C1(i0) = sum;
        }

        for (size_t i0 = 0; i0 < 3; i0++) {
            REQUIRE_THAT(C0(i0), Catch::Matchers::WithinRel(C1(i0), TestType{0.0001}));
        }
    }

    SECTION("3x5 <- 3x4x5 * 4x3x5") {
        auto C0 = create_tensor<TestType>("C0", 3, 5);
        auto C1 = create_tensor<TestType>("C1", 3, 5);
        // zero(C0);
        // zero(C1);
        auto A = create_random_tensor<TestType>("A", 3, 4, 5);
        auto B = create_random_tensor<TestType>("B", 4, 3, 5);

        // profile::push("einsum: 3x5 <- 3x4x5 * 4x3x5");
        REQUIRE_NOTHROW(einsum(Indices{i, k}, &C0, Indices{i, j, k}, A, Indices{j, i, k}, B));
        // profile::pop();

        // profile::push("hand  : 3x5 <- 3x4x5 * 4x3x5");
        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t k0 = 0; k0 < 5; k0++) {
                TestType sum{0};
                for (size_t j0 = 0; j0 < 4; j0++) {

                    sum += A(i0, j0, k0) * B(j0, i0, k0);
                }
                C1(i0, k0) = sum;
            }
        }
        // profile::pop();

        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t j0 = 0; j0 < 5; j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), TestType{0.0001}));
            }
        }
    }

    SECTION("3, l <- 3x4x5 * 4x3x5") {
        auto C0 = create_tensor<TestType>("C0", 3, 5);
        auto C1 = create_tensor<TestType>("C1", 3, 5);
        auto A  = create_random_tensor<TestType>("A", 3, 4, 5);
        auto B  = create_random_tensor<TestType>("B", 4, 3, 5);

        C0.zero();
        C1.zero();
        // profile::push("einsum: 3x5 <- 3x4x5 * 4x3x5");
        REQUIRE_NOTHROW(einsum(Indices{i, l}, &C0, Indices{i, j, k}, A, Indices{j, i, k}, B));
        // profile::pop();

        // profile::push("hand  : 3x5 <- 3x4x5 * 4x3x5");
        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t k0 = 0; k0 < 5; k0++) {
                for (size_t l0 = 0; l0 < 5; l0++) {
                    TestType sum{0};
                    for (size_t j0 = 0; j0 < 4; j0++) {

                        sum += A(i0, j0, k0) * B(j0, i0, k0);
                    }
                    C1(i0, l0) += sum;
                }
            }
        }
        // profile::pop();

        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t j0 = 0; j0 < 5; j0++) {
                // REQUIRE(C0(i0, j0) == C1(i0, j0));
                CHECK_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), TestType{0.0001}));
            }
        }
    }

    // profile::report();
    // profile::finalize();
}
