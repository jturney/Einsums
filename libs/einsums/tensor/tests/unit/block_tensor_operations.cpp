//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/tensor/block_tensor.hpp>
#include <einsums/testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("block tensor operations", "[tensor][block-tensor]", float, double) {
    auto A = create_block_tensor("A", 3, 3);
    auto B = create_block_tensor("B", 3, 3);

    A.zero();
    B.zero();

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

    A *= 2;
    A += 2;
    A /= 2;
    A -= 2;

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i == j) {
                CHECK(A(i, j) == 0.0);
            } else if (A.block_of(i) == A.block_of(j)) {
                CHECK(A(i, j) == -1.0);
            } else {
                CHECK(A(i, j) == 0.0);
            }
        }
    }

    A += B; // Swap here to avoid division by zero.
    B *= A;
    B /= A; // Either way there would be a division by zero. Do the addition first to not.
    B -= A;

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i == j) {
                CHECK(A(i, j) == 1.0);
                CHECK(B(i, j) == 0.0);
            } else if (A.block_of(i) == A.block_of(j)) {
                CHECK(A(i, j) == -1.0);
                CHECK(B(i, j) == 1.0);
            } else {
                CHECK(A(i, j) == 0.0);
                CHECK(B(i, j) == 0.0);
            }
        }
    }
}