//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("swap", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int N = 4;

    auto x = create_tensor<T>("x", N);
    auto y = create_tensor<T>("y", N);

    x(0) = 1.0;
    x(1) = 2.0;
    x(2) = 3.0;
    x(3) = 4.0;
    y(0) = 5.0;
    y(1) = 6.0;
    y(2) = 7.0;
    y(3) = 8.0;

    blas::swap<T>(N, x.data(), 1, y.data(), 1);

    // After swap, x should have y's original values and vice versa
    REQUIRE(x(0) == Catch::Approx(T{5.0}));
    REQUIRE(x(1) == Catch::Approx(T{6.0}));
    REQUIRE(x(2) == Catch::Approx(T{7.0}));
    REQUIRE(x(3) == Catch::Approx(T{8.0}));

    REQUIRE(y(0) == Catch::Approx(T{1.0}));
    REQUIRE(y(1) == Catch::Approx(T{2.0}));
    REQUIRE(y(2) == Catch::Approx(T{3.0}));
    REQUIRE(y(3) == Catch::Approx(T{4.0}));
}
