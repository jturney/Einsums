//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>

#include <catch2/catch_all.hpp>

using namespace einsums;

template <typename T>
void test_invert() {
    tensor A = create_tensor<T>("A", 3, 3);
    A(0, 0)  = 1.0;
    A(0, 1)  = 2.0;
    A(0, 2)  = 3.0;
    A(1, 0)  = 3.0;
    A(1, 1)  = 2.0;
    A(1, 2)  = 1.0;
    A(2, 0)  = 2.0;
    A(2, 1)  = 1.0;
    A(2, 2)  = 3.0;

    linear_algebra::invert(&A);

    CHECK_THAT(A.vector_data(), Catch::Matchers::Approx(
                                    vector_type<T>{-5.0 / 12, 0.25, 1.0 / 3.0, 7.0 / 12.0, 0.25, -2.0 / 3.0, 1.0 / 12.0, -0.25, 1.0 / 3.0})
                                    .margin(0.00001));
}

TEST_CASE("invert") {
    SECTION("float") {
        test_invert<float>();
    }
    SECTION("double") {
        test_invert<double>();
    }
}