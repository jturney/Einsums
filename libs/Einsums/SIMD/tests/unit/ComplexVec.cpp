//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Debugging.hpp>
#include <Einsums/SIMD/ComplexVec.hpp>
#include <Einsums/SIMD/Shuffle.hpp>

#include <complex>
#include <vector>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

TEMPLATE_TEST_CASE("CVec complex_broadcast and load round-trip", "[simd][complex]", float, double) {
    constexpr int          N = CVec<TestType>::complex_lanes;
    std::complex<TestType> val(3.0, 4.0);
    auto                   v = complex_broadcast(val);

    // Store and check
    std::vector<std::complex<TestType>> buf(N);
    complex_storeu(buf.data(), v);

    for (int i = 0; i < N; ++i) {
        CHECK(buf[i].real() == Catch::Approx(TestType(3.0)));
        CHECK(buf[i].imag() == Catch::Approx(TestType(4.0)));
    }
}

TEMPLATE_TEST_CASE("CVec complex_loadu / complex_storeu round-trip", "[simd][complex]", float, double) {
    constexpr int                       N = CVec<TestType>::complex_lanes;
    std::vector<std::complex<TestType>> src(N);
    for (int i = 0; i < N; ++i)
        src[i] = std::complex<TestType>(TestType(i), TestType(i + 10));

    auto v = complex_loadu(src.data());

    std::vector<std::complex<TestType>> dst(N);
    complex_storeu(dst.data(), v);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i].real() == Catch::Approx(src[i].real()));
        CHECK(dst[i].imag() == Catch::Approx(src[i].imag()));
    }
}

TEMPLATE_TEST_CASE("CVec conjugate", "[simd][complex]", float, double) {
    constexpr int                       N = CVec<TestType>::complex_lanes;
    std::vector<std::complex<TestType>> src(N);
    for (int i = 0; i < N; ++i)
        src[i] = std::complex<TestType>(TestType(i + 1), TestType(i + 2));

    auto v = complex_loadu(src.data());
    auto c = conjugate(v);

    std::vector<std::complex<TestType>> dst(N);
    complex_storeu(dst.data(), c);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i].real() == Catch::Approx(src[i].real()));
        CHECK(dst[i].imag() == Catch::Approx(-src[i].imag()));
    }
}

TEMPLATE_TEST_CASE("CVec complex_add", "[simd][complex]", float, double) {
    constexpr int          N = CVec<TestType>::complex_lanes;
    std::complex<TestType> a_val(2.0, 3.0);
    std::complex<TestType> b_val(1.0, 4.0);

    auto a = complex_broadcast(a_val);
    auto b = complex_broadcast(b_val);
    auto c = complex_add(a, b);

    std::vector<std::complex<TestType>> dst(N);
    complex_storeu(dst.data(), c);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i].real() == Catch::Approx(TestType(3.0)));
        CHECK(dst[i].imag() == Catch::Approx(TestType(7.0)));
    }
}

TEMPLATE_TEST_CASE("CVec complex_mul", "[simd][complex]", float, double) {
    constexpr int          N = CVec<TestType>::complex_lanes;
    std::complex<TestType> a_val(2.0, 3.0);
    std::complex<TestType> b_val(4.0, 5.0);

    auto expected = a_val * b_val; // (2*4 - 3*5) + i(2*5 + 3*4) = -7 + 22i

    auto a = complex_broadcast(a_val);
    auto b = complex_broadcast(b_val);
    auto c = complex_mul(a, b);

    std::vector<std::complex<TestType>> dst(N);
    complex_storeu(dst.data(), c);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i].real() == Catch::Approx(expected.real()));
        CHECK(dst[i].imag() == Catch::Approx(expected.imag()));
    }
}

TEMPLATE_TEST_CASE("CVec complex_gather / complex_scatter", "[simd][complex]", float, double) {
    constexpr int                       N      = CVec<TestType>::complex_lanes;
    constexpr int                       stride = 3;
    std::vector<std::complex<TestType>> src(N * stride, std::complex<TestType>(0, 0));

    for (int i = 0; i < N; ++i)
        src[i * stride] = std::complex<TestType>(TestType(i), TestType(i * 10));

    auto v = complex_gather(src.data(), std::ptrdiff_t(stride));

    std::vector<std::complex<TestType>> dst(N * stride, std::complex<TestType>(0, 0));
    complex_scatter(dst.data(), std::ptrdiff_t(stride), v);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i * stride].real() == Catch::Approx(src[i * stride].real()));
        CHECK(dst[i * stride].imag() == Catch::Approx(src[i * stride].imag()));
    }
}

TEMPLATE_TEST_CASE("complex_transpose_inplace correctness", "[simd][complex]", float, double) {
    constexpr int N = CVec<TestType>::complex_lanes;
    if constexpr (N <= 1) {
        SUCCEED("Transpose is a no-op for 1×1");
        return;
    }

    // Build N×N complex matrix: m[i][j] = (i*N + j) + i*(i*N + j + 100)
    std::vector<std::complex<TestType>> matrix(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            matrix[i * N + j] = std::complex<TestType>(TestType(i * N + j), TestType(i * N + j + 100));

    CVec<TestType> rows[8]; // NOLINT: large enough for any N
    for (int i = 0; i < N; ++i)
        rows[i] = complex_loadu(&matrix[i * N]);

    complex_transpose_inplace(rows);

    std::vector<std::complex<TestType>> result(N * N);
    for (int i = 0; i < N; ++i)
        complex_storeu(&result[i * N], rows[i]);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            auto expected = matrix[j * N + i]; // transposed
            INFO("i=" << i << " j=" << j);
            CHECK(result[i * N + j].real() == Catch::Approx(expected.real()));
            CHECK(result[i * N + j].imag() == Catch::Approx(expected.imag()));
        }
    }
}

TEMPLATE_TEST_CASE("CVec complex_sub", "[simd][complex]", float, double) {
    constexpr int          N = CVec<TestType>::complex_lanes;
    std::complex<TestType> a_val(5.0, 7.0);
    std::complex<TestType> b_val(2.0, 3.0);

    auto a = complex_broadcast(a_val);
    auto b = complex_broadcast(b_val);
    auto c = complex_sub(a, b);

    std::vector<std::complex<TestType>> dst(N);
    complex_storeu(dst.data(), c);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i].real() == Catch::Approx(TestType(3.0)));
        CHECK(dst[i].imag() == Catch::Approx(TestType(4.0)));
    }
}

TEMPLATE_TEST_CASE("CVec complex_scale (real scalar)", "[simd][complex]", float, double) {
    constexpr int          N = CVec<TestType>::complex_lanes;
    std::complex<TestType> val(3.0, 4.0);
    TestType               scalar = TestType(2.0);

    auto v = complex_broadcast(val);
    auto r = complex_scale(v, scalar);

    std::vector<std::complex<TestType>> dst(N);
    complex_storeu(dst.data(), r);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i].real() == Catch::Approx(TestType(6.0)));
        CHECK(dst[i].imag() == Catch::Approx(TestType(8.0)));
    }
}

TEMPLATE_TEST_CASE("CVec complex_fmadd: a*b + c", "[simd][complex]", float, double) {
    constexpr int          N = CVec<TestType>::complex_lanes;
    std::complex<TestType> a_val(1.0, 2.0);
    std::complex<TestType> b_val(3.0, 4.0);
    std::complex<TestType> c_val(10.0, 20.0);

    auto expected = a_val * b_val + c_val; // (1+2i)(3+4i) + (10+20i) = (-5+10i) + (10+20i) = (5+30i)

    auto a = complex_broadcast(a_val);
    auto b = complex_broadcast(b_val);
    auto c = complex_broadcast(c_val);
    auto r = complex_fmadd(a, b, c);

    std::vector<std::complex<TestType>> dst(N);
    complex_storeu(dst.data(), r);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i].real() == Catch::Approx(expected.real()));
        CHECK(dst[i].imag() == Catch::Approx(expected.imag()));
    }
}

TEMPLATE_TEST_CASE("CVec operator overloads", "[simd][complex]", float, double) {
    constexpr int          N = CVec<TestType>::complex_lanes;
    std::complex<TestType> a_val(2.0, 3.0);
    std::complex<TestType> b_val(4.0, 5.0);

    auto a = complex_broadcast(a_val);
    auto b = complex_broadcast(b_val);

    auto sum  = a + b;
    auto diff = a - b;
    auto prod = a * b;

    auto expected_sum  = a_val + b_val;
    auto expected_diff = a_val - b_val;
    auto expected_prod = a_val * b_val;

    std::vector<std::complex<TestType>> dst(N);

    complex_storeu(dst.data(), sum);
    for (int i = 0; i < N; ++i) {
        CHECK(dst[i].real() == Catch::Approx(expected_sum.real()));
        CHECK(dst[i].imag() == Catch::Approx(expected_sum.imag()));
    }

    complex_storeu(dst.data(), diff);
    for (int i = 0; i < N; ++i) {
        CHECK(dst[i].real() == Catch::Approx(expected_diff.real()));
        CHECK(dst[i].imag() == Catch::Approx(expected_diff.imag()));
    }

    complex_storeu(dst.data(), prod);
    for (int i = 0; i < N; ++i) {
        CHECK(dst[i].real() == Catch::Approx(expected_prod.real()));
        CHECK(dst[i].imag() == Catch::Approx(expected_prod.imag()));
    }
}
