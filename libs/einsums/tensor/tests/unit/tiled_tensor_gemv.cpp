//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/tensor/tiled_tensor.hpp>

#include <einsums/testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("TiledTensor GEMVs", "[tensor]", float, double) {

    tiled_tensor<TestType, 2> A("A", std::array{1, 0, 2});
    tensor<TestType, 1>       x("x", 3);
    tensor<TestType, 1>       y("y", 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((x.dim(0) == 3));
    REQUIRE((y.dim(0) == 3));

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(i, j) = 3 * i + j + 1;
        }
        x(i) = 11 * i + 11;
    }

    einsums::linear_algebra::gemv<false>(1.0, A, x, 0.0, &y);
    auto res = vector_type<TestType>{154.0, 352.0, 550.0};

    for (int i = 0; i < 3; i++) {
        CHECK(y(i) == res[i]);
    }

    einsums::linear_algebra::gemv<true>(1.0, A, x, 0.0, &y);
    res = vector_type<TestType>{330.0, 396.0, 462.0};
    for (int i = 0; i < 3; i++) {
        CHECK(y(i) == res[i]);
    }
}
