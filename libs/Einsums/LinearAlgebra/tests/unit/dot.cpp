//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/Testing/Layouts.hpp>

#include <array>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <complex>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("dot", "[linear-algebra]", float, double) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    constexpr int size = 10;

    SECTION("Rank 1 tensors") {
        Tensor<TestType, 1> A = create_random_tensor<TestType>("A", size);
        Tensor<TestType, 1> B = create_random_tensor<TestType>("B", size);

        TestType test{0.0};

        for (int i = 0; i < size; i++) {
            test += A(i) * B(i);
        }

        auto dot_res = dot(A, B);

        REQUIRE_THAT(dot_res, einsums::CheckWithinRel(test));
    }

    SECTION("Rank 1 tensor views") {
        Tensor<TestType, 2> A = create_random_tensor<TestType>("A", size, size);
        Tensor<TestType, 2> B = create_random_tensor<TestType>("B", size, size);

        for (int i = 0; i < size; i++) {
            auto A_view = A(All, i);
            auto B_view = B(i, All);

            TestType test{0.0};

            for (int j = 0; j < size; j++) {
                test += A(j, i) * B(i, j);
            }

            auto dot_res = dot(A_view, B_view);

            REQUIRE_THAT(dot_res, einsums::CheckWithinRel(test));
        }
    }

    SECTION("Rank 2 tensors") {
        Tensor<TestType, 2> A = create_random_tensor<TestType>("A", size, size);
        Tensor<TestType, 2> B = create_random_tensor<TestType>("B", size, size);

        TestType test{0.0};

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                test += A(i, j) * B(i, j);
            }
        }

        auto dot_res = dot(A, B);

        REQUIRE_THAT(dot_res, einsums::CheckWithinRel(test));
    }

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

        REQUIRE_THAT(dot_res, einsums::CheckWithinRel(test));
    }
}

TEMPLATE_TEST_CASE("Disk dot", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    config::set(option::WorkBufferSize, std::string("1024"));

    SECTION("Rank 1 Tensors") {
        constexpr int size = 2000;

        auto A = create_random_tensor<TestType>("A", size);
        auto B = create_random_tensor<TestType>("A", size);

        DiskTensor<TestType, 1> A_disk{fmt::format("/dot/rank1/{}/A", type_name<TestType>()), size},
            B_disk{fmt::format("/dot/rank1/{}/B", type_name<TestType>()), size};

        A_disk(All) = A;
        B_disk(All) = B;

        auto expected = einsums::linear_algebra::dot(A, B);
        auto got      = einsums::linear_algebra::dot(A_disk, B_disk);

        REQUIRE_THAT(got, einsums::CheckWithinRel(expected));
    }

    SECTION("Rank 2 Tensors") {
        constexpr size_t size1 = 100, size2 = 51;
        auto             A = create_random_tensor<TestType>("A", size1, size2);
        auto             B = create_random_tensor<TestType>("B", size1, size2);

        DiskTensor<TestType, 2> A_disk{fmt::format("/dot/rank2/{}/A", type_name<TestType>()), size1, size2},
            B_disk{fmt::format("/dot/rank2/{}/B", type_name<TestType>()), size1, size2};

        A_disk(All, All) = A;
        B_disk(All, All) = B;

        auto expected = einsums::linear_algebra::dot(A, B);
        auto got      = einsums::linear_algebra::dot(A_disk, B_disk);

        REQUIRE_THAT(got, einsums::CheckWithinRel(expected));
    }
}

namespace {

using einsums::testing::for_each_index;
using einsums::testing::for_each_layout_combination;
using einsums::testing::Layout;
using einsums::testing::layout_close;
using einsums::testing::layout_value;
using einsums::testing::logical_index;
using einsums::testing::shape_name;

/// Checks @p op(impls...) against the sum over every element of @p term(values...), on every
/// combination of layouts of @p Ts. Summation order differs between layouts, so the tolerance grows
/// with the element count and is relative to the sum of the terms' magnitudes.
template <typename R, typename... Ts, typename Op, typename Term>
void check_every_layout_reduction(Op op, Term term) {
    for_each_layout_combination<Ts...>([&](std::vector<size_t> const &dims, Layout<Ts>... operands) {
        std::string names;
        ((names += " " + operands.name + ","), ...);
        INFO("shape " << shape_name(dims) << ", layouts" << names);

        using Wide = std::conditional_t<IsComplexV<R>, std::complex<double>, double>;
        Wide   sum{0};
        double mag  = 0;
        size_t size = 0;
        for_each_index(dims, [&](std::vector<size_t> const &index) {
            size_t const k = logical_index(dims, index);
            Wide const   t = [&]<size_t... I>(std::index_sequence<I...>) {
                return term(static_cast<Wide>(layout_value<Ts>(static_cast<unsigned>(I + 1), k))...);
            }(std::index_sequence_for<Ts...>{});
            sum += t;
            mag += std::abs(t);
            size++;
        });
        R const want = static_cast<R>(sum);

        R const got = op(operands.impl()...);
        INFO("got " << got << ", want " << want);
        CHECK(layout_close(got, want, mag, 8.0 + 2.0 * size));
    });
}

} // namespace

// dot picks one of three paths by layout: one vector call when the operands are single vectors with the
// same strides, loops over vector calls when they share strides but are not one vector, and an element
// walk otherwise. Every combination of layouts meets here, with extent-1 axes, rank 0 and an empty extent.
TEMPLATE_TEST_CASE("dot on every combination of layouts", "[linear-algebra][dot][layouts]", float, double, std::complex<float>,
                   std::complex<double>) {
    using T = TestType;
    SECTION("dot") {
        check_every_layout_reduction<T, T, T>([](auto const &a, auto const &b) { return linear_algebra::detail::impl_dot(a, b); },
                                              [](auto a, auto b) { return a * b; });
    }
    SECTION("true_dot") {
        check_every_layout_reduction<T, T, T>([](auto const &a, auto const &b) { return linear_algebra::detail::impl_true_dot(a, b); },
                                              [](auto a, auto b) {
                                                  if constexpr (IsComplexV<T>) {
                                                      return std::conj(a) * b;
                                                  } else {
                                                      return a * b;
                                                  }
                                              });
    }
    SECTION("three-operand dot") {
        check_every_layout_reduction<T, T, T, T>(
            [](auto const &a, auto const &b, auto const &c) { return linear_algebra::detail::impl_dot(a, b, c); },
            [](auto a, auto b, auto c) { return a * b * c; });
    }
}

// A tensor that is one vector is one call to the vector routine whatever its size, so its dot is the
// BLAS dot of the same data to the bit. The run walk sent stretches shorter than its cutoff to an
// element loop, which rounds differently: a 4-element float dot drifted by an ulp from numpy's, and
// two DIIS implementations meant to agree bitwise stopped agreeing.
TEMPLATE_TEST_CASE("dot of a single vector is the BLAS dot to the bit", "[linear-algebra][dot]", float, double, std::complex<float>,
                   std::complex<double>) {
    using T = TestType;
    for (size_t const n : {1, 3, 4, 15, 16, 17, 64}) {
        INFO("n = " << n);
        std::vector<T> a(n), b(n);
        for (size_t i = 0; i < n; i++) {
            a[i] = einsums::testing::layout_value<T>(1, i);
            b[i] = einsums::testing::layout_value<T>(2, i);
        }
        einsums::detail::TensorImpl<T> const A(a.data(), std::array{n}), B(b.data(), std::array{n});
        CHECK(linear_algebra::detail::impl_dot(A, B) == blas::dot(static_cast<blas::int_t>(n), a.data(), 1, b.data(), 1));
        if constexpr (IsComplexV<T>) {
            CHECK(linear_algebra::detail::impl_true_dot(A, B) == blas::dotc(static_cast<blas::int_t>(n), a.data(), 1, b.data(), 1));
        }
    }
}
