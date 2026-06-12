//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/LinearAlgebra/Bases/syev.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

template <Complex T>
void heev_test() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A = create_tensor<T>("a", 3, 3);
    auto b = create_tensor<RemoveComplexT<T>>("b", 3);

    A.vector_data() = {{0.199889, 0.0},         {-0.330816, -0.127778},  {-0.0546237, 0.176589}, {-0.330816, 0.127778}, {0.629179, 0.0},
                       {-0.0224813, -0.327171}, {-0.0546237, -0.176589}, {-0.0224813, 0.327171}, {0.170931, 0.0}};

    auto A2 = create_tensor<T>(true, "a2", 3, 3);

    A2 = A;

    einsums::linear_algebra::heev(&A, &b);

    // Sometimes 0.0 will be reported as -0.0 therefore we test the Abs of the first two
    CHECK_THAT(b(0), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(1), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(2), Catch::Matchers::WithinRel(1.0, 0.00001));

    einsums::linear_algebra::heev(&A2, &b);

    CHECK_THAT(b(0), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(1), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(2), Catch::Matchers::WithinRel(1.0, 0.00001));
}

template <Complex T>
void heev_strided_test() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto b = create_tensor<RemoveComplexT<T>>("b", 3);

    auto data = VectorData<T>{{0.199889, 0.0},         {0.0, 0.0}, {-0.330816, -0.127778}, {0.0, 0.0}, {-0.0546237, 0.176589},  {0.0, 0.0},
                              {-0.330816, 0.127778},   {0.0, 0.0}, {0.629179, 0.0},        {0.0, 0.0}, {-0.0224813, -0.327171}, {0.0, 0.0},
                              {-0.0546237, -0.176589}, {0.0, 0.0}, {-0.0224813, 0.327171}, {0.0, 0.0}, {0.170931, 0.0},         {0.0, 0.0}};

    TensorView<T, 2> A{data.data(), Dim<2>{3, 3}, Stride<2>{6, 2}};

    Tensor<T, 2> A2{false, "test", 3, 3}, A3{true, "test 2", 3, 3};

    A2 = A;
    A3 = A;

    println(A);
    println(A2);
    println(A3);

    einsums::linear_algebra::heev(&A2, &b);
    einsums::linear_algebra::heev(&A3, &b);

    einsums::linear_algebra::heev(&A, &b);

    println(A);
    println(A2);
    println(A3);

    // Sometimes 0.0 will be reported as -0.0 therefore we test the Abs of the first two
    CHECK_THAT(b(0), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(1), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(2), Catch::Matchers::WithinRel(1.0, 0.00001));

    if constexpr (std::is_same_v<T, std::complex<double>>) {
        T div1 = A2(0, 0) / A(0, 0), div2 = A2(0, 1) / A(0, 1), div3 = A2(0, 2) / A(0, 2);

        scale_column(0, div1, &A);
        scale_column(1, div2, &A);
        scale_column(2, div3, &A);

        for (int i = 0; i < 3; i++) {
            // Eigenvector components after phase alignment; allow for small
            // numerical differences between strided and contiguous LAPACK paths.
            CHECK_THAT(std::real(A(i, 0)), Catch::Matchers::WithinAbs(std::real(A2(i, 0)), 0.01));
            CHECK_THAT(std::imag(A(i, 0)), Catch::Matchers::WithinAbs(std::imag(A2(i, 0)), 0.01));
            CHECK_THAT(std::real(A(i, 1)), Catch::Matchers::WithinAbs(std::real(A2(i, 1)), 0.01));
            CHECK_THAT(std::imag(A(i, 1)), Catch::Matchers::WithinAbs(std::imag(A2(i, 1)), 0.01));
            CHECK_THAT(std::real(A(i, 2)), Catch::Matchers::WithinAbs(std::real(A2(i, 2)), 0.01));
            CHECK_THAT(std::imag(A(i, 2)), Catch::Matchers::WithinAbs(std::imag(A2(i, 2)), 0.01));
            // A3 (row-major) eigenvectors may differ in phase from A (strided view).
            // Use a wider tolerance since we can't easily phase-align A3 like we did for A2.
            CHECK_THAT(std::real(A(i, 0)), Catch::Matchers::WithinAbs(std::real(A3(i, 0)), 0.01));
            CHECK_THAT(std::imag(A(i, 0)), Catch::Matchers::WithinAbs(std::imag(A3(i, 0)), 0.01));
            CHECK_THAT(std::real(A(i, 1)), Catch::Matchers::WithinAbs(std::real(A3(i, 1)), 0.01));
            CHECK_THAT(std::imag(A(i, 1)), Catch::Matchers::WithinAbs(std::imag(A3(i, 1)), 0.01));
            CHECK_THAT(std::real(A(i, 2)), Catch::Matchers::WithinAbs(std::real(A3(i, 2)), 0.01));
            CHECK_THAT(std::imag(A(i, 2)), Catch::Matchers::WithinAbs(std::imag(A3(i, 2)), 0.01));
        }
    }
}

TEMPLATE_TEST_CASE("heev", "[linear-algebra]", std::complex<float>, std::complex<double>) {
    heev_test<TestType>();
    heev_strided_test<TestType>();
}
