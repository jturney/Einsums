//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>
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

namespace {

/// The direct product's element against one computed here. A complex product's error is bounded by
/// the magnitudes of its factors, not by its own: when the two terms of a real or imaginary part
/// nearly cancel, that part is far smaller than the rounding it carries, and a bound relative to it
/// fails for a kernel that rounds a fused multiply-add once where this reference rounds twice.
template <typename T>
void check_product(T got, T want, T a, T b) {
    double const eps = std::numeric_limits<RemoveComplexT<T>>::epsilon();
    // The largest prefactor these cases use is 1 + i, of magnitude sqrt(2).
    REQUIRE(std::abs(got - want) <= 8 * eps * 1.5 * std::abs(a) * std::abs(b));
}

} // namespace

TEMPLATE_TEST_CASE("direct product", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
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
                C_test(i) = TestType{1.0, 1.0} * A(i) * B(i);
            }
            direct_product(TestType{1.0, 1.0}, A, B, TestType{0.0}, &C);
        } else {

            for (int i = 0; i < size; i++) {
                C_test(i) = A(i) * B(i);
            }
            direct_product(TestType{1.0}, A, B, TestType{0.0}, &C);
        }

        for (int i = 0; i < size; i++) {
            check_product(C(i), C_test(i), A(i), B(i));
        }
    }

    SECTION("Rank 1 tensors small") {
        constexpr int size = 4;

        Tensor<TestType, 1> A = create_random_tensor<TestType>("A", size);
        Tensor<TestType, 1> B = create_random_tensor<TestType>("B", size);

        Tensor<TestType, 1> C      = create_tensor<TestType>("C", size);
        Tensor<TestType, 1> C_test = create_tensor<TestType>("C", size);

        for (int i = 0; i < size; i++) {
            C_test(i) = A(i) * B(i);
        }

        direct_product(TestType{1.0}, A, B, TestType{0.0}, &C);

        for (int i = 0; i < size; i++) {
            check_product(C(i), C_test(i), A(i), B(i));
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
            C_test(i) = A(i, 0) * B(i, 0);
        }

        direct_product(TestType{1.0}, A_view, B_view, TestType{0.0}, &C_view);

        for (int i = 0; i < size; i++) {
            check_product(C(i, 0), C_test(i), A(i, 0), B(i, 0));
        }
    }

    SECTION("Rank 1 tensor views small") {
        constexpr int size = 4;

        Tensor<TestType, 2> A = create_random_tensor<TestType>(true, "A", size, 2);
        Tensor<TestType, 2> B = create_random_tensor<TestType>(true, "B", size, 2);

        Tensor<TestType, 2> C(true, "C", size, 2);
        Tensor<TestType, 1> C_test("C", size);

        auto A_view = A(All, 0);
        auto B_view = B(All, 0);
        auto C_view = C(All, 0);

        for (int i = 0; i < size; i++) {
            C_test(i) = A(i, 0) * B(i, 0);
        }

        direct_product(TestType{1.0}, A_view, B_view, TestType{0.0}, &C_view);

        for (int i = 0; i < size; i++) {
            check_product(C(i, 0), C_test(i), A(i, 0), B(i, 0));
        }
    }

    // SECTION("Rank 2 tensors") {
    //     constexpr int size = 10;
    //     Tensor<TestType, 2> A = create_random_tensor<TestType>("A", size, size);
    //     Tensor<TestType, 2> B = create_random_tensor<TestType>("B", size, size);

    //     TestType test{0.0};

    //     for (int i = 0; i < size; i++) {
    //         for (int j = 0; j < size; j++) {
    //             test += A(i, j) * B(i, j);
    //         }
    //     }

    //     auto dot_res = dot(A, B);

    //     REQUIRE_THAT(dot_res, einsums::WithinStrict(test, TestType{100000.0}));
    // }

    // SECTION("Rank 2 tensor views") {
    //     Tensor<TestType, 2> A = create_random_tensor<TestType>("A", size, size);
    //     Tensor<TestType, 2> B = create_random_tensor<TestType>("B", size, size);

    //     for(int i = 0; i < size; i++) {
    //         auto A_view = A(AllT(), Range{i, i + 1});
    //         auto B_view = B(AllT(), Range{i, i + 1});

    //         TestType test{0.0};

    //         for(int j = 0; j < size; j++) {
    //             test += A(j, i) * B(i, j);
    //         }

    //         auto dot_res = dot(A_view, B_view);

    //         REQUIRE(std::abs(dot_res - test) < static_cast<TestType>(EINSUMS_ZERO));
    //     }
    // }
}

TEST_CASE("mixed dots", "[linear-algebra]") {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    constexpr int size = 10;

    SECTION("Rank 1 tensors") {
        Tensor<float, 1>  A = create_random_tensor<float>("A", size);
        Tensor<double, 1> B = create_random_tensor<double>("B", size);

        double test{0.0};

        for (int i = 0; i < size; i++) {
            test += A(i) * B(i);
        }

        auto dot_res = dot(A, B);

        REQUIRE_THAT(dot_res, einsums::WithinStrict(test, double{100000.0}));
    }
}

TEMPLATE_TEST_CASE("Disk direct products", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    config::set(option::WorkBufferSize, std::string("1024"));

    SECTION("Rank 1 tensors") {
        constexpr int size = 2000;

        Tensor<TestType, 1> A = create_random_tensor<TestType>("A", size);
        Tensor<TestType, 1> B = create_random_tensor<TestType>("B", size);

        Tensor<TestType, 1> C = create_tensor<TestType>("C", size);
        C.zero();
        Tensor<TestType, 1> C_test = create_tensor<TestType>("C", size);

        DiskTensor<TestType, 1> A_disk{fmt::format("/dirprod/rank1/{}/A", type_name<TestType>()), size};
        DiskTensor<TestType, 1> B_disk{fmt::format("/dirprod/rank1/{}/B", type_name<TestType>()), size};
        DiskTensor<TestType, 1> C_disk{fmt::format("/dirprod/rank1/{}/C", type_name<TestType>()), size};

        A_disk(All).get() = A;
        B_disk(All).get() = B;

        if constexpr (IsComplexV<TestType>) {
            for (int i = 0; i < size; i++) {
                C_test(i) = TestType{1.0, 1.0} * A(i) * B(i);
            }
            direct_product(TestType{1.0, 1.0}, A_disk, B_disk, TestType{0.0}, &C_disk);
        } else {

            for (int i = 0; i < size; i++) {
                C_test(i) = A(i) * B(i);
            }
            direct_product(TestType{1.0}, A_disk, B_disk, TestType{0.0}, &C_disk);
        }

        C = C_disk(All).get();

        for (int i = 0; i < size; i++) {
            check_product(C(i), C_test(i), A(i), B(i));
        }
    }
}

// A dense output beside a strided view has different strides, so these take the general fallback.
// It used to recurse over the axes in index order, which for column-major tensors walks the largest
// stride innermost (about 8 ns per element on a DF-MP2 denominator); it now walks the output's own
// order a run at a time. Either way the result must be the index loop's.
TEMPLATE_TEST_CASE("elementwise ops on a dense tensor beside a strided view", "[linear-algebra]", float, double, std::complex<float>,
                   std::complex<double>) {
    using namespace einsums::linear_algebra;
    using T            = TestType;
    auto         P     = create_random_tensor<T>("P", 6, 7, 8);
    auto         Q     = create_random_tensor<T>("Q", 5, 7, 9);
    auto const   A     = P(Range{1, 5}, All, Range{2, 6});
    auto const   B     = Q(Range{0, 4}, All, Range{3, 7});
    auto         C     = create_random_tensor<T>("C", 4, 7, 4);
    T const      alpha = T{0.75};
    T const      beta  = T{-1.25};
    double const tol   = 100.0 * std::numeric_limits<RemoveComplexT<T>>::epsilon();
    Tensor<T, 3> C_ref = C;

    auto const check = [&]() {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 7; ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    REQUIRE(std::abs(C(i, j, k) - C_ref(i, j, k)) <= tol * (1.0 + std::abs(C_ref(i, j, k))));
                }
            }
        }
    };
    auto const each = [&](auto &&f) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 7; ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    f(i, j, k);
                }
            }
        }
    };

    SECTION("direct product") {
        each([&](size_t i, size_t j, size_t k) { C_ref(i, j, k) = beta * C(i, j, k) + alpha * A(i, j, k) * B(i, j, k); });
        direct_product(alpha, A, B, beta, &C);
        check();
    }
    SECTION("direct division") {
        each([&](size_t i, size_t j, size_t k) { C_ref(i, j, k) = beta * C(i, j, k) + alpha * A(i, j, k) / B(i, j, k); });
        direct_division(alpha, A, B, beta, &C);
        check();
    }
    SECTION("axpy") {
        each([&](size_t i, size_t j, size_t k) { C_ref(i, j, k) = C(i, j, k) + alpha * A(i, j, k); });
        axpy(alpha, A, &C);
        check();
    }
}

// A zero beta discards C on every layout path: the dense one, the loops over blocks of a shared
// strided layout, and the fully strided one. Each scaled C by zero before adding the product, so a C
// carved from reused memory kept its NaN wherever the layout sent it.
TEMPLATE_TEST_CASE("direct product with a zero beta discards C on every layout path", "[linear-algebra]", float, double,
                   std::complex<float>, std::complex<double>) {
    using namespace einsums::linear_algebra;
    using T          = TestType;
    auto const   nan = std::numeric_limits<RemoveComplexT<T>>::quiet_NaN();
    T const      poison{nan};
    T const      alpha = T{0.75};
    double const tol   = 100.0 * std::numeric_limits<RemoveComplexT<T>>::epsilon();

    auto const check = [&](auto const &A, auto const &B, auto const &C) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 7; ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    T const want = alpha * A(i, j, k) * B(i, j, k);
                    REQUIRE(std::abs(C(i, j, k) - want) <= tol * (1.0 + std::abs(want)));
                }
            }
        }
    };

    SECTION("dense") {
        auto const A = create_random_tensor<T>("A", 4, 7, 4);
        auto const B = create_random_tensor<T>("B", 4, 7, 4);
        auto       C = create_tensor<T>("C", 4, 7, 4);
        C.set_all(poison);
        direct_product(alpha, A, B, T{0}, &C);
        check(A, B, C);
    }
    SECTION("blocks of a shared strided layout") {
        auto P = create_random_tensor<T>("P", 6, 7, 8);
        auto Q = create_random_tensor<T>("Q", 6, 7, 8);
        auto R = create_tensor<T>("R", 6, 7, 8);
        R.set_all(poison);
        auto const A = P(Range{1, 5}, All, Range{2, 6});
        auto const B = Q(Range{1, 5}, All, Range{2, 6});
        auto       C = R(Range{1, 5}, All, Range{2, 6});
        direct_product(alpha, A, B, T{0}, &C);
        check(A, B, C);
    }
    SECTION("fully strided") {
        auto       P = create_random_tensor<T>("P", 6, 7, 8);
        auto const A = P(Range{1, 5}, All, Range{2, 6});
        auto const B = create_random_tensor<T>("B", 4, 7, 4);
        auto       C = create_tensor<T>("C", 4, 7, 4);
        C.set_all(poison);
        direct_product(alpha, A, B, T{0}, &C);
        check(A, B, C);
    }
}

// direct_product picks one of three paths by layout: one pass when the operands are single vectors with the
// same strides, loops over passes when they share strides but are not one vector, and an element walk
// otherwise. Every combination of layouts meets here, with extent-1 axes, rank 0 and an empty extent.
// A zero beta never reads C: a C filled with NaN still comes out finite.
TEMPLATE_TEST_CASE("direct_product on every combination of layouts", "[linear-algebra][direct_prod][layouts]", float, double,
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
            linear_algebra::detail::impl_direct_product(alpha, A_impl, B_impl, beta, &C_impl);

            size_t      wrong = 0;
            std::string first;
            einsums::testing::for_each_index(dims, [&](std::vector<size_t> const &index) {
                size_t const k = logical_index(dims, index);
                Wide const   a = layout_value<T>(1, k), b = layout_value<T>(2, k), c = layout_value<T>(3, k);
                Wide const   product = Wide(alpha) * (a * b);
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
