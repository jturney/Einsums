//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>

#include <span>

#include <Einsums/Testing.hpp>

using namespace einsums;

template <NotComplex T>
void getrf_and_getri_test() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto                      A = create_tensor<T>("A", 4, 4);
    BufferVector<blas::int_t> pivot(4);

    A.vector_data() = {1.80, 2.88, 2.05, -0.89, 5.25, -2.95, -0.95, -3.80, 1.58, -2.69, -2.90, -1.04, -1.11, -0.66, -0.59, 0.80};

    auto ret = einsums::linear_algebra::getrf(&A, &pivot);

    if (ret != 0) {
        REQUIRE_THROWS(einsums::linear_algebra::getri(&A, pivot));
    } else {
        einsums::linear_algebra::getri(&A, pivot);

        CHECK_THAT(A.vector_data(),
                   Catch::Matchers::Approx(std::vector<T>{1.77199817, 0.57569082, 0.08432537, 4.81550236, -0.11746607, -0.44561501,
                                                          0.41136261, -1.71258093, 0.17985639, 0.45266204, -0.66756530, 1.48240005,
                                                          2.49438204, 0.76497689, -0.03595380, 7.61190029})
                       .margin(0.01));
    }
}

TEMPLATE_TEST_CASE("getrf_getri", "[linear-algebra]", float, double) {
    getrf_and_getri_test<TestType>();
}
