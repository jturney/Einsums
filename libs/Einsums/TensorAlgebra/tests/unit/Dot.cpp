//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

EINSUMS_TEMPLATE_TEST_CASE("Einsum Dot Product", "[tensor-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;
    using namespace einsums::linear_algebra;

    size_t i_{10}, j_{10}, a_{10}, b_{10};

    tensor_algebra::detail::AlgorithmChoice alg_choice;

    SECTION("1") {
        auto                A = create_random_tensor<TestType>("A", i_);
        auto                B = create_random_tensor<TestType>("B", i_);
        Tensor<TestType, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i}, A, Indices{i}, B, &alg_choice);
        REQUIRE(alg_choice == einsums::tensor_algebra::detail::DOT);

        if constexpr (!einsums::IsComplexV<TestType>) {
            REQUIRE_THAT(C_obtained, Catch::Matchers::WithinAbsMatcher(C_expected, 0.0001));
        } else {
            REQUIRE_THAT(((TestType)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), 0.0001));
            REQUIRE_THAT(((TestType)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), 0.0001));
        }
    }

    SECTION("2") {
        auto                A = create_random_tensor<TestType>("A", i_, j_);
        auto                B = create_random_tensor<TestType>("B", i_, j_);
        Tensor<TestType, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i, j}, A, Indices{i, j}, B, &alg_choice);
        REQUIRE(alg_choice == einsums::tensor_algebra::detail::DOT);

        if constexpr (!einsums::IsComplexV<TestType>) {
            REQUIRE_THAT(C_obtained, Catch::Matchers::WithinAbsMatcher(C_expected, 0.0001));
        } else {
            REQUIRE_THAT(((TestType)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), 0.0001));
            REQUIRE_THAT(((TestType)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), 0.0001));
        }
    }

    SECTION("3") {
        auto                A = create_random_tensor<TestType>("A", i_, j_, a_);
        auto                B = create_random_tensor<TestType>("B", i_, j_, a_);
        Tensor<TestType, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i, j, a}, A, Indices{i, j, a}, B, &alg_choice);
        REQUIRE(alg_choice == einsums::tensor_algebra::detail::DOT);

        if constexpr (!einsums::IsComplexV<TestType>) {
            REQUIRE_THAT(C_obtained, Catch::Matchers::WithinAbsMatcher(C_expected, 0.0001));
        } else {
            REQUIRE_THAT(((TestType)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), 0.0001));
            REQUIRE_THAT(((TestType)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), 0.0001));
        }
    }

    SECTION("4") {
        auto                A = create_random_tensor<TestType>("A", i_, j_, a_, b_);
        auto                B = create_random_tensor<TestType>("B", i_, j_, a_, b_);
        Tensor<TestType, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i, j, a, b}, A, Indices{i, j, a, b}, B, &alg_choice);
        REQUIRE(alg_choice == einsums::tensor_algebra::detail::DOT);

        if constexpr (!einsums::IsComplexV<TestType>) {
            REQUIRE_THAT(C_obtained, Catch::Matchers::WithinAbsMatcher(C_expected, 0.0001));
        } else {
            REQUIRE_THAT(((TestType)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), 0.0001));
            REQUIRE_THAT(((TestType)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), 0.0001));
        }
    }
}

TEMPLATE_TEST_CASE("Dot TensorView and Tensor", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    tensor_algebra::detail::AlgorithmChoice alg_choice;

    auto const i_ = 10, j_ = 10, k_ = 10, l_ = 2;

    auto     A  = create_random_tensor<TestType>("A", i_, k_);
    auto     B  = create_random_tensor<TestType>("B", l_, k_);
    TestType C  = TestType{0.0};
    TestType C0 = TestType{0.0};

    auto A_view = A(Range{0, l_}, All); // (l_, k_)

    einsum(Indices{}, &C, Indices{l, k}, A_view, Indices{l, k}, B, &alg_choice);
    REQUIRE(alg_choice == einsums::tensor_algebra::detail::DOT);

    for (size_t l = 0; l < l_; l++) {
        for (size_t k = 0; k < k_; k++) {
            C0 += A(l, k) * B(l, k);
        }
    }

    REQUIRE_THAT(C, CheckWithinRel(C0, 0.0001));
}
