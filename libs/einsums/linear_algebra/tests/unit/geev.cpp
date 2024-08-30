//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/testing.hpp>

using namespace einsums;

template <NotComplex T>
void geev_test() {

    auto a  = create_tensor<T>("a", 5, 5);
    auto w  = create_tensor<add_complex_t<T>>("w", 5);
    auto vl = create_tensor<T>("vl", 5, 5);
    auto vr = create_tensor<T>("vr", 5, 5);

    a.vector_data() = vector_type<T>{-1.01f, 0.86f,  -4.60f, 3.31f, -4.81f, 3.98f,  0.53f, -7.04f, 5.29f,  3.55f,  3.30f, 8.26f, -3.89f,
                                     8.20f,  -1.51f, 4.43f,  4.96f, -7.66f, -7.33f, 6.18f, 7.31f,  -6.43f, -6.16f, 2.47f, 5.58f};

    linear_algebra::geev(&a, &w, &vl, &vr);

    //    (0): (    2.85813284  +    10.76274967i)
    //    (1): (    2.85813284  +   -10.76274967i)
    //    (2): (   -0.68667454  +     4.70426130i)
    //    (3): (   -0.68667454  +    -4.70426130i)
    //    (4): (  -10.46291637  +     0.00000000i)

    CHECK_THAT(w(0).real(), Catch::Matchers::WithinRel(2.85813284, 0.001));
    CHECK_THAT(w(1).real(), Catch::Matchers::WithinRel(2.85813284, 0.001));
    CHECK_THAT(w(2).real(), Catch::Matchers::WithinRel(-0.68667454, 0.001));
    CHECK_THAT(w(3).real(), Catch::Matchers::WithinRel(-0.68667454, 0.001));
    CHECK_THAT(w(4).real(), Catch::Matchers::WithinRel(-10.46291637, 0.001));

    CHECK_THAT(w(0).imag(), Catch::Matchers::WithinRel(10.76274967, 0.001));
    CHECK_THAT(w(1).imag(), Catch::Matchers::WithinRel(-10.76274967, 0.001));
    CHECK_THAT(w(2).imag(), Catch::Matchers::WithinRel(4.70426130, 0.001));
    CHECK_THAT(w(3).imag(), Catch::Matchers::WithinRel(-4.70426130, 0.001));
    CHECK_THAT(w(4).imag(), Catch::Matchers::WithinRel(0.00, 0.001));
}

template <Complex T>
void geev_test() {

    auto a  = create_tensor<T>("a", 4, 4);
    auto w  = create_tensor<T>("w", 4);
    auto vl = create_tensor<T>("vl", 4, 4);
    auto vr = create_tensor<T>("vr", 4, 4);

    a.vector_data() =
        vector_type<T>{{-3.84f, 2.25f},  {-8.94f, -4.75f}, {8.95f, -6.53f},  {-9.87f, 4.82f},  {-0.66f, 0.83f},  {-4.40f, -3.82f},
                       {-3.50f, -4.26f}, {-3.15f, 7.36f},  {-3.99f, -4.73f}, {-5.88f, -6.60f}, {-3.36f, -0.40f}, {-0.75f, 5.23f},
                       {7.74f, 4.18f},   {3.66f, -7.53f},  {2.58f, 3.60f},   {4.59f, 5.41f}};

    linear_algebra::geev(&a, &w, &vl, &vr);

    CHECK_THAT(w(0).real(), Catch::Matchers::WithinRel(-9.4298, 0.001));
    CHECK_THAT(w(1).real(), Catch::Matchers::WithinRel(-3.4419, 0.001));
    CHECK_THAT(w(2).real(), Catch::Matchers::WithinRel(0.1056, 0.001));
    CHECK_THAT(w(3).real(), Catch::Matchers::WithinRel(5.7562, 0.001));

    CHECK_THAT(w(0).imag(), Catch::Matchers::WithinRel(-12.9833, 0.001));
    CHECK_THAT(w(1).imag(), Catch::Matchers::WithinRel(12.6897, 0.001));
    CHECK_THAT(w(2).imag(), Catch::Matchers::WithinRel(-3.3950, 0.001));
    CHECK_THAT(w(3).imag(), Catch::Matchers::WithinRel(7.1286, 0.001));
}

TEMPLATE_TEST_CASE("geev", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    geev_test<TestType>();
}
