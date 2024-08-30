//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/testing.hpp>

using namespace einsums;

template <Complex T>
void heev_test() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A = create_tensor<T>("a", 3, 3);
    auto b = create_tensor<remove_complex_t<T>>("b", 3);

    A.vector_data() =
        vector_type<T>{{0.199889, 0.0},       {-0.330816, -0.127778},  {-0.0546237, 0.176589}, {-0.330816, 0.127778}, {0.629179, 0.0},
                       {-0.224813, 0.327171}, {-0.0546237, -0.176589}, {-0.0224813, 0.327171}, {0.170931, 0.0}};

    einsums::linear_algebra::heev(&A, &b);

    // Sometimes 0.0 will be reported as -0.0 therefore we test the Abs of the first two
    CHECK_THAT(b(0), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(1), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(2), Catch::Matchers::WithinRel(1.0, 0.00001));
}

TEMPLATE_TEST_CASE("heev", "[linear-algebra]", std::complex<float>, std::complex<double>) {
    heev_test<TestType>();
}
