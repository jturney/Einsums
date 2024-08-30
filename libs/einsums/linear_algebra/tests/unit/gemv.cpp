//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/modules/tensor_utilities.hpp>

#include <catch2/catch_all.hpp>

using namespace einsums;

template <typename T>
void test_gemv() {
    tensor A = create_tensor<T>("A", 3, 3);
    tensor x = create_tensor<T>("x", 3);
    tensor y = create_tensor<T>("y", 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((x.dim(0) == 3));
    REQUIRE((y.dim(0) == 3));

    A.vector_data() = vector_type<T>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    x.vector_data() = vector_type<T>{11.0, 22.0, 33.0};

    einsums::linear_algebra::gemv<false>(1.0, A, x, 0.0, &y);
    CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(vector_type<T>{154.0, 352.0, 550.0}));

    einsums::linear_algebra::gemv<true>(1.0, A, x, 0.0, &y);
    CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(vector_type<T>{330.0, 396.0, 462.0}));
}

TEST_CASE("gemv") {
    SECTION("float") {
        test_gemv<float>();
    }
    SECTION("double") {
        test_gemv<double>();
    }
}

template <NotComplex T>
void gemv_test2() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A  = create_incremented_tensor<T>("A", 3, 3);
    auto b  = create_incremented_tensor<T>("b", 3);
    auto fy = create_incremented_tensor<T>("y", 3);
    auto ty = create_incremented_tensor<T>("y", 3);

    // Perform basic matrix-vector
    einsums::linear_algebra::gemv<false>(1.0, A, b, 0.0, &fy);
    einsums::linear_algebra::gemv<true>(1.0, A, b, 0.0, &ty);

    REQUIRE(fy(0) == 5.0);
    REQUIRE(fy(1) == 14.0);
    REQUIRE(fy(2) == 23.0);

    REQUIRE(ty(0) == 15.0);
    REQUIRE(ty(1) == 18.0);
    REQUIRE(ty(2) == 21.0);
}

template <Complex T>
void gemv_test2() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A  = create_incremented_tensor<T>("A", 3, 3);
    auto b  = create_incremented_tensor<T>("b", 3);
    auto fy = create_incremented_tensor<T>("y", 3);
    auto ty = create_incremented_tensor<T>("y", 3);

    // Perform basic matrix-vector
    einsums::linear_algebra::gemv<false>(T{1.0, 1.0}, A, b, T{0.0, 0.0}, &fy);
    einsums::linear_algebra::gemv<true>(T{1.0, 1.0}, A, b, T{0.0, 0.0}, &ty);

    REQUIRE(fy(0) == T{-10, 10});
    REQUIRE(fy(1) == T{-28, 28});
    REQUIRE(fy(2) == T{-46, 46});

    REQUIRE(ty(0) == T{-30, 30});
    REQUIRE(ty(1) == T{-36, 36});
    REQUIRE(ty(2) == T{-42, 42});
}

TEST_CASE("gemv2") {
    SECTION("double") {
        gemv_test2<double>();
    }
    SECTION("float") {
        gemv_test2<float>();
    }
    SECTION("complex<double>") {
        gemv_test2<std::complex<double>>();
    }
    SECTION("complex<float>") {
        gemv_test2<std::complex<float>>();
    }
}
