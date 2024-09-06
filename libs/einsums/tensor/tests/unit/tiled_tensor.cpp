//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/tensor/tiled_tensor.hpp>

#include <einsums/testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("creation", "[tensor][tiled-tensor]", float, double) {
    tiled_tensor<TestType, 2> A("A", std::array{1, 0, 2});
    tiled_tensor<TestType, 2> B("B", std::array{1, 0, 2});
    tiled_tensor<TestType, 2> C("C", std::array{1, 0, 2});

    REQUIRE(A.dim(0) == 3);
    REQUIRE(A.dim(1) == 3);
    REQUIRE(B.dim(0) == 3);
    REQUIRE(B.dim(1) == 3);
    REQUIRE(A.tile_offset(0)[0] == 0);
    REQUIRE(A.tile_offset(0)[1] == 1);
    REQUIRE(A.tile_offset(0)[2] == 1);
    REQUIRE(B.tile_offset(0)[0] == 0);
    REQUIRE(B.tile_offset(0)[1] == 1);
    REQUIRE(B.tile_offset(0)[2] == 1);
    REQUIRE(A.tile_offset(1)[0] == 0);
    REQUIRE(A.tile_offset(1)[1] == 1);
    REQUIRE(A.tile_offset(1)[2] == 1);
    REQUIRE(B.tile_offset(1)[0] == 0);
    REQUIRE(B.tile_offset(1)[1] == 1);
    REQUIRE(B.tile_offset(1)[2] == 1);

    A.zero();
    B.zero();

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(fabs(A(i, j)) < 1e-11);
            CHECK(fabs(B(i, j)) < 1e-11);
        }
    }

    // Set A and B to identity
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;

    B(0, 0) = 1.0;
    B(1, 1) = 1.0;
    B(2, 2) = 1.0;

    // Perform basic matrix multiplication
    einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {
                CHECK(A(i, j) == 1.0);
                CHECK(B(i, j) == 1.0);
                CHECK(C(i, j) == 1.0);
            } else {
                CHECK(A(i, j) == 0.0);
                CHECK(B(i, j) == 0.0);
                CHECK(C(i, j) == 0.0);
            }
        }
    }
}