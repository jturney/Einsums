//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/tensor/block_tensor.hpp>
#include <einsums/testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("creation", "[tensor][block-tensor]", double, float) {
    auto                      A = create_block_tensor<TestType>("A", 3, 3);
    auto                      B = create_block_tensor<TestType>("B", 3, 3);
    block_tensor<TestType, 2> C;

    REQUIRE((A.dim(0) == 6 && A.dim(1) == 6));
    REQUIRE((B.dim(0) == 6 && B.dim(1) == 6));

    A.zero();
    B.zero();

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            CHECK(A(i, j) == 0.0);
            CHECK(B(i, j) == 0.0);
        }
    }

    // Set A and B to identity
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;
    A(3, 3) = 1.0;
    A(4, 4) = 1.0;
    A(5, 5) = 1.0;

    B(0, 0) = 1.0;
    B(1, 1) = 1.0;
    B(2, 2) = 1.0;
    B(3, 3) = 1.0;
    B(4, 4) = 1.0;
    B(5, 5) = 1.0;

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i == j) {
                CHECK(A(i, j) == 1.0);
                CHECK(B(i, j) == 1.0);
            } else {
                CHECK(A(i, j) == 0.0);
                CHECK(B(i, j) == 0.0);
            }
        }
    }

    // Set up C using a different method.
    C.push_block(tensor<TestType, 2>("block2", 3, 3));
    C.insert_block(0, tensor<TestType, 2>("block1", 3, 3));
    C.set_name("C");
    C.zero();

    // Perform basic matrix multiplication
    einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i == j) {
                CHECK(C(i, j) == 1.0);
            } else {
                CHECK(C(i, j) == 0.0);
            }
        }
    }

    // Check by blocks.
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {
                CHECK(A[0](i, j) == 1.0);
                CHECK(B[0](i, j) == 1.0);
                CHECK(A[1](i, j) == 1.0);
                CHECK(B[1](i, j) == 1.0);
                CHECK(C["block1"](i, j) == 1.0);
                CHECK(C["block2"](i, j) == 1.0);
            } else {
                CHECK(A[0](i, j) == 0.0);
                CHECK(B[0](i, j) == 0.0);
                CHECK(A[1](i, j) == 0.0);
                CHECK(B[1](i, j) == 0.0);
                CHECK(C["block1"](i, j) == 0.0);
                CHECK(C["block2"](i, j) == 0.0);
            }
        }
    }

    CHECK_THROWS(C["block3"]);

    A.set_all(2.0);

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i / 3 == j / 3) {
                REQUIRE(A.block_of(i) == A.block_of(j));
            } else {
                REQUIRE(A.block_of(i) != A.block_of(j));
            }
            if (A.block_of(i) == A.block_of(j)) {
                CHECK(A(i, j) == 2.0);
            } else {
                CHECK(A(i, j) == 0.0);
            }
        }
    }
}