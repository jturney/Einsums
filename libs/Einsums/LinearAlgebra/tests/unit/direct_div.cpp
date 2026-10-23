//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/Testing/Layouts.hpp>

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <complex>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

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

// direct_division picks one of three paths by layout: one pass when the operands are single vectors with the
// same strides, loops over passes when they share strides but are not one vector, and an element walk
// otherwise. Every combination of layouts meets here, with extent-1 axes, rank 0 and an empty extent.
// A zero beta never reads C: a C filled with NaN still comes out finite.
TEMPLATE_TEST_CASE("direct_division on every combination of layouts", "[linear-algebra][direct_div][layouts]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T    = TestType;
    using R    = RemoveComplexT<T>;
    using Wide = std::conditional_t<IsComplexV<T>, std::complex<double>, double>;
    using einsums::testing::Layout;
    using einsums::testing::layout_value;
    using einsums::testing::logical_index;

    auto const scalar = [](double re, double im) {
        if constexpr (IsComplexV<T>) {
            return T(static_cast<R>(re), static_cast<R>(im));
        } else {
            return static_cast<T>(re);
        }
    };
    T const alpha = scalar(1.5, -0.5);

    for (T const beta : {scalar(-0.75, 0.25), T(0)}) {
        INFO("beta " << beta);
        einsums::testing::for_each_layout_combination<T, T, T>([&](std::vector<size_t> const &dims, Layout<T> A, Layout<T> B, Layout<T> C) {
            INFO("shape " << einsums::testing::shape_name(dims) << ", A " << A.name << ", B " << B.name << ", C " << C.name);
            if (beta == T(0)) {
                einsums::testing::for_each_index(
                    dims, [&](std::vector<size_t> const &index) { C.at(index) = T(std::numeric_limits<R>::quiet_NaN()); });
            }
            auto const A_impl = A.impl(), B_impl = B.impl();
            auto       C_impl = C.impl();
            linear_algebra::detail::impl_direct_division(alpha, A_impl, B_impl, beta, &C_impl);

            size_t      wrong = 0;
            std::string first;
            einsums::testing::for_each_index(dims, [&](std::vector<size_t> const &index) {
                size_t const k = logical_index(dims, index);
                Wide const   a = layout_value<T>(1, k), b = layout_value<T>(2, k), c = layout_value<T>(3, k);
                Wide const   product = Wide(alpha) * (a / b);
                Wide const   want    = beta == T(0) ? product : Wide(beta) * c + product;
                double const scale   = std::abs(product) + (beta == T(0) ? 0.0 : std::abs(Wide(beta) * c));
                if (!(std::abs(Wide(C.at(index)) - want) <= 8 * std::numeric_limits<R>::epsilon() * scale) && wrong++ == 0) {
                    first = fmt::format("element {} is {}, want {}", k, C.at(index), static_cast<T>(want));
                }
            });
            INFO(first);
            CHECK(wrong == 0);
            CHECK(C.gaps_untouched());
        });
    }
}
