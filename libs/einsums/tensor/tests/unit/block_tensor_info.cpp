//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/tensor/block_tensor.hpp>
#include <einsums/testing.hpp>

using namespace einsums;

TEST_CASE("block tensor info", "[tensor][block-tensor]") {
    auto A = create_block_tensor<double>("A", 3, 3);

    for (int i = 0; i < 6; i++) {
        CHECK(A.block_of(i) == i / 3);
    }

    CHECK_THROWS(A.block_of(6));
    CHECK_THROWS(A.block_of(-1));
}