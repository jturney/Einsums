//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Debugging.hpp>
#include <Einsums/SIMD/ComplexVec.hpp>
#include <Einsums/SIMD/Shuffle.hpp>

#include <algorithm>
#include <complex>
#include <memory>
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

// Every lane holds a different pair, so a product that mixes lanes, or swaps
// which lane subtracts and which adds, cannot pass. Small integers keep every
// product and sum exact, so fused and unfused rungs must agree bitwise.
TEMPLATE_TEST_CASE("CVec complex_mul: distinct lanes", "[simd][complex]", float, double) {
    constexpr int                       N = CVec<TestType>::complex_lanes;
    std::vector<std::complex<TestType>> a(N), b(N), dst(N);
    for (int i = 0; i < N; ++i) {
        a[i] = {TestType(i + 1), TestType(-2 * i + 3)};
        b[i] = {TestType(3 * i - 4), TestType(i + 5)};
    }

    complex_storeu(dst.data(), complex_mul(complex_loadu(a.data()), complex_loadu(b.data())));

    for (int i = 0; i < N; ++i) {
        INFO("lane " << i);
        CHECK(dst[i] == a[i] * b[i]);
    }
}

// complex_fmadd is built on complex_mul; the same distinct-lane check for it.
TEMPLATE_TEST_CASE("CVec complex_fmadd: distinct lanes", "[simd][complex]", float, double) {
    constexpr int                       N = CVec<TestType>::complex_lanes;
    std::vector<std::complex<TestType>> a(N), b(N), c(N), dst(N);
    for (int i = 0; i < N; ++i) {
        a[i] = {TestType(2 * i - 1), TestType(i + 2)};
        b[i] = {TestType(i - 3), TestType(-i - 1)};
        c[i] = {TestType(7 - i), TestType(3 * i)};
    }

    complex_storeu(dst.data(), complex_fmadd(complex_loadu(a.data()), complex_loadu(b.data()), complex_loadu(c.data())));

    for (int i = 0; i < N; ++i) {
        INFO("lane " << i);
        CHECK(dst[i] == a[i] * b[i] + c[i]);
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

TEMPLATE_TEST_CASE("complex partial load and store at every count", "[simd][complex][partial]", float, double) {
    using C          = std::complex<TestType>;
    constexpr int  N = CVec<TestType>::complex_lanes;
    std::vector<C> src(N);
    for (int i = 0; i < N; ++i) {
        src[i] = C(static_cast<TestType>(i + 1), static_cast<TestType>(-(i + 1)));
    }
    for (std::size_t n = 0; n <= static_cast<std::size_t>(N) + 1; ++n) {
        INFO("n = " << n);
        C loaded[N];
        complex_storeu(loaded, complex_loadu_partial(src.data(), n));
        for (int i = 0; i < N; ++i) {
            CHECK(loaded[i] == (static_cast<std::size_t>(i) < n ? src[i] : C(0, 0)));
        }
        std::vector<C> dst(N + 2, C(99, 99));
        complex_storeu_partial(dst.data(), complex_loadu(src.data()), n);
        for (int i = 0; i < N + 2; ++i) {
            CHECK(dst[i] == (static_cast<std::size_t>(i) < std::min(n, static_cast<std::size_t>(N)) ? src[i] : C(99, 99)));
        }
    }
    // Exact-size allocations: a partial access past n values is a heap overflow the sanitizers see.
    for (std::size_t n = 1; n < static_cast<std::size_t>(N); ++n) {
        auto const in  = std::make_unique<C[]>(n);
        auto const out = std::make_unique<C[]>(n);
        for (std::size_t i = 0; i < n; ++i) {
            in[i] = src[i];
        }
        complex_storeu_partial(out.get(), complex_loadu_partial(in.get(), n), n);
        for (std::size_t i = 0; i < n; ++i) {
            CHECK(out[i] == in[i]);
        }
    }
}

TEMPLATE_TEST_CASE("complex_reduce_add sums real and imaginary parts apart", "[simd][complex][reduce]", float, double) {
    using C         = std::complex<TestType>;
    constexpr int N = CVec<TestType>::complex_lanes;
    // Integer values, so every partial sum is exact and the fold order cannot show.
    C in[N];
    C expected(0, 0);
    for (int i = 0; i < N; ++i) {
        in[i] = C(static_cast<TestType>(i + 1), static_cast<TestType>(-10 * (i + 1)));
        expected += in[i];
    }
    CHECK(complex_reduce_add(complex_loadu(in)) == expected);
}
