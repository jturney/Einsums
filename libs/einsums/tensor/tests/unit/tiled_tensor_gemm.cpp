//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/tensor/tiled_tensor.hpp>

#include <einsums/testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("TiledTensor GEMMs", "[tensor]", float, double) {
    using namespace einsums;

    tiled_tensor<TestType, 2> A("A", std::array{1, 0, 2});
    tiled_tensor<TestType, 2> B("B", std::array{1, 0, 2});
    tiled_tensor<TestType, 2> C("C", std::array{1, 0, 2});

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((B.dim(0) == 3 && B.dim(1) == 3));
    REQUIRE((C.dim(0) == 3 && C.dim(1) == 3));

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(i, j) = 3 * i + j + 1;
            B(i, j) = 33 * i + 11 * j + 11;
        }
    }

    einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
    auto res = vector_type<TestType>{330.0, 396.0, 462.0, 726.0, 891.0, 1056.0, 1122.0, 1386.0, 1650.0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(C(i, j) == res[3 * i + j]);
        }
    }

    einsums::linear_algebra::gemm<true, false>(1.0, A, B, 0.0, &C);
    res = vector_type<TestType>{726.0, 858.0, 990.0, 858.0, 1023.0, 1188.0, 990.0, 1188.0, 1386.0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(C(i, j) == res[3 * i + j]);
        }
    }

    einsums::linear_algebra::gemm<false, true>(1.0, A, B, 0.0, &C);
    res = vector_type<TestType>{154.0, 352.0, 550.0, 352.0, 847.0, 1342.0, 550.0, 1342.0, 2134.0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(C(i, j) == res[3 * i + j]);
        }
    }

    einsums::linear_algebra::gemm<true, true>(1.0, A, B, 0.0, &C);
    res = vector_type<TestType>{330.0, 726.0, 1122.0, 396.0, 891.0, 1386.0, 462.0, 1056.0, 1650.0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(C(i, j) == res[3 * i + j]);
        }
    }
}
