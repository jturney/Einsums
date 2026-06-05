//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

// Element-wise (Hadamard) division: C = alpha * (A ./ B) + beta * C. Counterpart
// to direct_product. Covers dense tensors and views (the path used for CC
// amplitude denominators); Block/Tiled/Disk overloads are intentionally absent.
TEMPLATE_TEST_CASE("direct division", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums::linear_algebra;

    SECTION("Rank 1 tensors") {
        constexpr int size = 21;

        Tensor<TestType, 1> A = create_random_tensor<TestType>("A", size);
        Tensor<TestType, 1> B = create_random_tensor<TestType>("B", size);

        Tensor<TestType, 1> C = create_tensor<TestType>("C", size);
        C.zero();
        Tensor<TestType, 1> C_test = create_tensor<TestType>("C", size);

        if constexpr (IsComplexV<TestType>) {
            for (int i = 0; i < size; i++) {
                C_test(i) = TestType{1.0, 1.0} * A(i) / B(i);
            }
            direct_division(TestType{1.0, 1.0}, A, B, TestType{0.0}, &C);
        } else {
            for (int i = 0; i < size; i++) {
                C_test(i) = A(i) / B(i);
            }
            direct_division(TestType{1.0}, A, B, TestType{0.0}, &C);
        }

        for (int i = 0; i < size; i++) {
            if constexpr (IsComplexV<TestType>) {
                REQUIRE_THAT(std::real(C_test(i)), Catch::Matchers::WithinRel(std::real(C(i))));
                REQUIRE_THAT(std::imag(C_test(i)), Catch::Matchers::WithinRel(std::imag(C(i))));
            } else {
                REQUIRE_THAT(C_test(i), Catch::Matchers::WithinRel(C(i)));
            }
        }
    }

    SECTION("beta accumulate") {
        constexpr int size = 16;

        Tensor<TestType, 1> A = create_random_tensor<TestType>("A", size);
        Tensor<TestType, 1> B = create_random_tensor<TestType>("B", size);
        Tensor<TestType, 1> C = create_random_tensor<TestType>("C", size);
        Tensor<TestType, 1> C_test("Cref", size);

        for (int i = 0; i < size; i++) {
            C_test(i) = TestType{2.0} * C(i) + TestType{3.0} * A(i) / B(i);
        }
        direct_division(TestType{3.0}, A, B, TestType{2.0}, &C);

        for (int i = 0; i < size; i++) {
            if constexpr (IsComplexV<TestType>) {
                REQUIRE_THAT(std::real(C_test(i)), Catch::Matchers::WithinRel(std::real(C(i))));
                REQUIRE_THAT(std::imag(C_test(i)), Catch::Matchers::WithinRel(std::imag(C(i))));
            } else {
                REQUIRE_THAT(C_test(i), Catch::Matchers::WithinRel(C(i)));
            }
        }
    }

    SECTION("Rank 1 tensor views") {
        constexpr int size = 21;

        Tensor<TestType, 2> A = create_random_tensor<TestType>(true, "A", size, 1);
        Tensor<TestType, 2> B = create_random_tensor<TestType>(true, "B", size, 2);

        Tensor<TestType, 2> C(true, "C", size, 3);
        Tensor<TestType, 1> C_test("C", size);

        auto A_view = A(All, 0);
        auto B_view = B(All, 0);
        auto C_view = C(All, 0);

        for (int i = 0; i < size; i++) {
            C_test(i) = A(i, 0) / B(i, 0);
        }

        direct_division(TestType{1.0}, A_view, B_view, TestType{0.0}, &C_view);

        for (int i = 0; i < size; i++) {
            if constexpr (IsComplexV<TestType>) {
                REQUIRE_THAT(std::real(C_test(i)), Catch::Matchers::WithinRel(std::real(C(i, 0))));
                REQUIRE_THAT(std::imag(C_test(i)), Catch::Matchers::WithinRel(std::imag(C(i, 0))));
            } else {
                REQUIRE_THAT(C_test(i), Catch::Matchers::WithinRel(C(i, 0)));
            }
        }
    }
}
