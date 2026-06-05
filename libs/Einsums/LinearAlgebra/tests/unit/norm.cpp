//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("Norms", "[linear-algebra]", float, double) {
    using T = TestType;

    auto A = create_tensor<T>("a", 9, 9);

    auto temp = std::vector<T>{1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,  0.0,  0.0, 4.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0,  6.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,  8.0,  1.0, 1.0, 1.0, 1.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 11.0, 12.0};

    A.vector_data() = temp;

    T result = linear_algebra::norm(linear_algebra::Norm::ONE, A);

    if (A.impl().is_row_major()) {
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(15.0, 0.1));
    } else {
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(33.0, 0.1));
    }

    result = linear_algebra::norm(linear_algebra::Norm::INFTY, A);

    if (A.impl().is_row_major()) {
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(33.0, 0.1));
    } else {
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(15.0, 0.1));
    }

    result = linear_algebra::norm(linear_algebra::Norm::FROBENIUS, A);

    REQUIRE_THAT(result, Catch::Matchers::WithinRel(25.92296279363144, 0.001));

    result = linear_algebra::norm(linear_algebra::Norm::MAXABS, A);

    REQUIRE_THAT(result, Catch::Matchers::WithinRel(12.0, 0.0001));

    result = linear_algebra::norm(linear_algebra::Norm::TWO, A);

    REQUIRE_THAT(result, Catch::Matchers::WithinRel(19.46610381488656, 0.0001));
}
