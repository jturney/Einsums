//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Debugging.hpp>
#include <Einsums/SIMD/Gather.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <vector>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

TEMPLATE_TEST_CASE("gather with stride 1", "[simd]", float, double) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> data(N);
    for (int i = 0; i < N; ++i)
        data[i] = TestType(i + 1);

    auto v = gather(data.data(), std::ptrdiff_t(1));

    for (int i = 0; i < N; ++i) {
        CHECK(v[i] == Catch::Approx(TestType(i + 1)));
    }
}

TEMPLATE_TEST_CASE("gather with stride 2", "[simd]", float, double) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> data(N * 2);
    for (int i = 0; i < static_cast<int>(data.size()); ++i)
        data[i] = TestType(i);

    auto v = gather(data.data(), std::ptrdiff_t(2));

    for (int i = 0; i < N; ++i) {
        CHECK(v[i] == Catch::Approx(TestType(i * 2)));
    }
}

TEMPLATE_TEST_CASE("gather with stride 3", "[simd]", float, double) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> data(N * 3);
    for (int i = 0; i < static_cast<int>(data.size()); ++i)
        data[i] = TestType(i);

    auto v = gather(data.data(), std::ptrdiff_t(3));

    for (int i = 0; i < N; ++i) {
        CHECK(v[i] == Catch::Approx(TestType(i * 3)));
    }
}

TEMPLATE_TEST_CASE("gather with large stride", "[simd]", float, double) {
    constexpr int         N      = Vec<TestType>::lanes;
    constexpr int         stride = 17;
    std::vector<TestType> data(N * stride);
    for (int i = 0; i < static_cast<int>(data.size()); ++i)
        data[i] = TestType(i * 0.1);

    auto v = gather(data.data(), std::ptrdiff_t(stride));

    for (int i = 0; i < N; ++i) {
        CHECK(v[i] == Catch::Approx(TestType(i * stride * 0.1)));
    }
}

TEMPLATE_TEST_CASE("scatter with stride 1", "[simd]", float, double) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> dst(N, TestType(0));
    auto                  v = broadcast(TestType(99.0));

    scatter(dst.data(), std::ptrdiff_t(1), v);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i] == Catch::Approx(TestType(99.0)));
    }
}

TEMPLATE_TEST_CASE("scatter with stride 2", "[simd]", float, double) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> dst(N * 2, TestType(0));

    // Build a vec with known values
    alignas(native_alignment) TestType src[N]; // NOLINT
    for (int i = 0; i < N; ++i)
        src[i] = TestType(i + 10);
    auto v = loadu(src);

    scatter(dst.data(), std::ptrdiff_t(2), v);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i * 2] == Catch::Approx(TestType(i + 10)));
    }
}

TEMPLATE_TEST_CASE("gather then scatter round-trip", "[simd]", float, double) {
    constexpr int         N      = Vec<TestType>::lanes;
    constexpr int         stride = 5;
    std::vector<TestType> src(N * stride, TestType(0));
    std::vector<TestType> dst(N * stride, TestType(0));

    // Fill source at strided positions
    for (int i = 0; i < N; ++i)
        src[i * stride] = TestType(i * 3.14);

    auto v = gather(src.data(), std::ptrdiff_t(stride));
    scatter(dst.data(), std::ptrdiff_t(stride), v);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i * stride] == Catch::Approx(src[i * stride]));
    }
}

// ===========================================================================
// gather_fixed / scatter_fixed (compile-time stride)
// ===========================================================================

TEMPLATE_TEST_CASE("gather_fixed<1> matches loadu", "[simd]", float, double) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> data(N);
    for (int i = 0; i < N; ++i)
        data[i] = TestType(i + 1);

    auto v = gather_fixed<1>(data.data());

    for (int i = 0; i < N; ++i) {
        CHECK(v[i] == Catch::Approx(TestType(i + 1)));
    }
}

TEMPLATE_TEST_CASE("gather_fixed<2>", "[simd]", float, double) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> data(N * 2);
    for (int i = 0; i < static_cast<int>(data.size()); ++i)
        data[i] = TestType(i);

    auto v = gather_fixed<2>(data.data());

    for (int i = 0; i < N; ++i) {
        CHECK(v[i] == Catch::Approx(TestType(i * 2)));
    }
}

TEMPLATE_TEST_CASE("gather_fixed<3>", "[simd]", float, double) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> data(N * 3);
    for (int i = 0; i < static_cast<int>(data.size()); ++i)
        data[i] = TestType(i);

    auto v = gather_fixed<3>(data.data());

    for (int i = 0; i < N; ++i) {
        CHECK(v[i] == Catch::Approx(TestType(i * 3)));
    }
}

TEMPLATE_TEST_CASE("scatter_fixed<1> matches storeu", "[simd]", float, double) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> dst(N, TestType(0));
    auto                  v = broadcast(TestType(42.0));

    scatter_fixed<1>(dst.data(), v);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i] == Catch::Approx(TestType(42.0)));
    }
}

TEMPLATE_TEST_CASE("gather_fixed / scatter_fixed round-trip", "[simd]", float, double) {
    constexpr int         N      = Vec<TestType>::lanes;
    constexpr int         stride = 4;
    std::vector<TestType> src(N * stride, TestType(0));
    std::vector<TestType> dst(N * stride, TestType(0));

    for (int i = 0; i < N; ++i)
        src[i * stride] = TestType(i * 2.5);

    auto v = gather_fixed<stride>(src.data());
    scatter_fixed<stride>(dst.data(), v);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i * stride] == Catch::Approx(src[i * stride]));
    }
}
