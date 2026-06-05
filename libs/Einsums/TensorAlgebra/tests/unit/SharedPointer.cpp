//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("shared_ptr", "[tensor_algebra]", float, double) { //, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    SECTION("C") {
        auto   C0 = std::make_shared<Tensor<TestType, 2>>("C0", 3, 3);
        Tensor C1 = create_tensor<TestType>("C1", 3, 3);
        Tensor A  = create_random_tensor<TestType>("A", 3, 5);
        Tensor B  = create_random_tensor<TestType>("B", 5, 3);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT(C0->operator()(i0, j0), CheckWithinRel(C1(i0, j0), 0.001));
            }
        }
    }

    SECTION("A") {
        Tensor C0 = create_tensor<TestType>("C0", 3, 3);
        Tensor C1 = create_tensor<TestType>("C1", 3, 3);
        auto   A  = std::make_shared<Tensor<TestType, 2>>(create_random_tensor<TestType>("A", 3, 5));
        Tensor B  = create_random_tensor<TestType>("B", 5, 3);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, *A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), CheckWithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("B") {
        Tensor C0 = create_tensor<TestType>("C0", 3, 3);
        Tensor C1 = create_tensor<TestType>("C1", 3, 3);
        Tensor A  = create_random_tensor<TestType>("A", 3, 5);
        auto   B  = std::make_shared<Tensor<TestType, 2>>(create_random_tensor<TestType>("B", 5, 3));

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), CheckWithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("AB") {
        Tensor C0 = create_tensor<TestType>("C0", 3, 3);
        Tensor C1 = create_tensor<TestType>("C1", 3, 3);
        auto   A  = std::make_shared<Tensor<TestType, 2>>(create_random_tensor<TestType>("A", 3, 5));
        auto   B  = std::make_shared<Tensor<TestType, 2>>(create_random_tensor<TestType>("B", 5, 3));

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, *A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), CheckWithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("CA") {
        auto   C0 = std::make_shared<Tensor<TestType, 2>>("C0", 3, 3);
        Tensor C1 = create_tensor<TestType>("C1", 3, 3);
        auto   A  = std::make_shared<Tensor<TestType, 2>>(create_random_tensor<TestType>("A", 3, 5));
        Tensor B  = create_random_tensor<TestType>("B", 5, 3);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, *A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT(C0->operator()(i0, j0), CheckWithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("CB") {
        auto   C0 = std::make_shared<Tensor<TestType, 2>>("C0", 3, 3);
        Tensor C1 = create_tensor<TestType>("C1", 3, 3);
        Tensor A  = create_random_tensor<TestType>("A", 3, 5);
        auto   B  = std::make_shared<Tensor<TestType, 2>>(create_random_tensor<TestType>("B", 5, 3));

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT(C0->operator()(i0, j0), CheckWithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("CAB") {
        auto   C0 = std::make_shared<Tensor<TestType, 2>>("C0", 3, 3);
        Tensor C1 = create_tensor<TestType>("C1", 3, 3);
        auto   A  = std::make_shared<Tensor<TestType, 2>>(create_random_tensor<TestType>("A", 3, 5));
        auto   B  = std::make_shared<Tensor<TestType, 2>>(create_random_tensor<TestType>("B", 5, 3));

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, *A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT(C0->operator()(i0, j0), CheckWithinRel(C1(i0, j0), RemoveComplexT<TestType>{0.0001}));
            }
        }
    }

    SECTION("explicit prefactors") {
        // Verify the with-prefactors overload works: 2.0 * C += 0.5 * A * B
        Tensor C0 = create_tensor<TestType>("C0", 3, 3);
        Tensor C1 = create_tensor<TestType>("C1", 3, 3);
        auto   A  = std::make_shared<Tensor<TestType, 2>>(create_random_tensor<TestType>("A", 3, 5));
        auto   B  = std::make_shared<Tensor<TestType, 2>>(create_random_tensor<TestType>("B", 5, 3));

        zero(C0);
        zero(C1);

        REQUIRE_NOTHROW(einsum(TestType{2.0}, Indices{i, j}, &C0, TestType{0.5}, Indices{i, k}, A, Indices{k, j}, B));
        // C1 = 0.5 * (*A) * (*B)  (C1 was zero, so 2.0 * 0 + 0.5 * A * B)
        linear_algebra::gemm<false, false>(0.5, *A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), CheckWithinRel(C1(i0, j0), RemoveComplexT<TestType>{0.001}));
            }
        }
    }

    SECTION("algorithm_choice") {
        Tensor C0 = create_tensor<TestType>("C0", 3, 3);
        auto   A  = std::make_shared<Tensor<TestType, 2>>(create_random_tensor<TestType>("A", 3, 5));
        auto   B  = std::make_shared<Tensor<TestType, 2>>(create_random_tensor<TestType>("B", 5, 3));

        tensor_algebra::detail::AlgorithmChoice alg_choice = tensor_algebra::detail::INDETERMINATE;
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B, &alg_choice));
        REQUIRE(alg_choice == tensor_algebra::detail::GEMM);
    }
}
