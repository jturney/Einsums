//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("invert", "[linear-algebra]", float, double) {
    SECTION("Invert") {
        Tensor A = create_tensor<TestType>(true, "A", 3, 3);
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

        CHECK_THAT(A.vector_data(), Catch::Matchers::Approx(std::vector<TestType>{-5.0 / 12, 0.25, 1.0 / 3.0, 7.0 / 12.0, 0.25, -2.0 / 3.0,
                                                                                  1.0 / 12.0, -0.25, 1.0 / 3.0})
                                        .margin(0.00001));
    }
    SECTION("Strided") {
        BufferVector<TestType>  A_data(18);
        TensorView<TestType, 2> A{A_data.data(), Dim{3, 3}, Stride{2, 6}};
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(0, 2) = 3.0;
        A(1, 0) = 3.0;
        A(1, 1) = 2.0;
        A(1, 2) = 1.0;
        A(2, 0) = 2.0;
        A(2, 1) = 1.0;
        A(2, 2) = 3.0;

        Tensor A_test        = create_tensor<TestType>(true, "A test", 3, 3);
        A_test.vector_data() = {-5.0 / 12, 0.25, 1.0 / 3.0, 7.0 / 12.0, 0.25, -2.0 / 3.0, 1.0 / 12.0, -0.25, 1.0 / 3.0};

        linear_algebra::invert(&A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK_THAT(A(i, j), Catch::Matchers::WithinRel(A_test(i, j)));
            }
        }
    }
}