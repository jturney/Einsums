//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Debugging.hpp>
#include <Einsums/SIMD/Convert.hpp>
#include <Einsums/SIMD/Gather.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__linux__)
#    include <sys/mman.h>
#endif

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

// ─── Index gather ─────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("gather by an index vector reads one table entry per lane", "[simd][gather]", float, double) {
    using T          = TestType;
    using I          = gather_index_t<T>;
    constexpr int  L = Vec<T>::lanes;
    std::vector<T> table(1009);
    for (size_t i = 0; i < table.size(); ++i)
        table[i] = T(0.5) * static_cast<T>(i) - T(17);

    // Out of order, repeated, and both ends of the table.
    for (int round = 0; round < 4; ++round) {
        I at[L];
        for (int i = 0; i < L; ++i)
            at[i] = static_cast<I>((round * 389 + (L - i) * 211) % 1009);
        at[0] = round % 2 ? 0 : 1008;
        if (L > 2)
            at[L - 1] = at[1];
        T out[L];
        storeu(out, gather(table.data(), loadu(at)));
        for (int i = 0; i < L; ++i) {
            INFO("lane " << i << ": index " << at[i]);
            CHECK(out[i] == table[static_cast<size_t>(at[i])]);
        }
    }
}

TEMPLATE_TEST_CASE("a table lookup composes floor, convert and gather", "[simd][gather]", float, double) {
    // The shape of a tabulated function evaluated per lane: the grid point below x, then the entry
    // there. Every intermediate stays in vector registers.
    using T           = TestType;
    using I           = gather_index_t<T>;
    constexpr int  L  = Vec<T>::lanes;
    T const        dx = T(0.125);
    std::vector<T> grid(400);
    for (size_t i = 0; i < grid.size(); ++i)
        grid[i] = std::exp(-static_cast<T>(i) * dx);

    T x[L];
    for (int i = 0; i < L; ++i)
        x[i] = T(0.37) + static_cast<T>(i) * T(3.3);
    Vec<T> const vx  = loadu(x);
    Vec<I> const idx = convert<I>(floor(mul(vx, broadcast(T(1) / dx))));
    T            out[L];
    storeu(out, gather(grid.data(), idx));
    for (int i = 0; i < L; ++i) {
        INFO("lane " << i << ": x = " << x[i]);
        CHECK(out[i] == grid[static_cast<size_t>(std::floor(x[i] / dx))]);
    }
}

#if defined(__linux__)
// Pins the fix for strides whose lane offsets overflow int32: the 32-bit-index hardware gather and
// scatter used to build the offsets as int and wrap. The reservation is mostly untouched address
// space; only the pages the lanes land on are ever backed.
TEST_CASE("float gather and scatter reach lanes past a 2^31-element offset", "[simd][gather]") {
    constexpr int        L      = Vec<float>::lanes;
    std::ptrdiff_t const stride = (std::ptrdiff_t{1} << 29) + 3;
    size_t const         bytes  = (static_cast<size_t>(L - 1) * static_cast<size_t>(stride) + 1) * sizeof(float);
    void                *region = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (region == MAP_FAILED) {
        SKIP("could not reserve " << bytes << " bytes of address space");
    }
    auto *base = static_cast<float *>(region);
    for (int i = 0; i < L; ++i)
        base[i * stride] = static_cast<float>(i + 1);

    float out[L];
    storeu(out, gather(base, stride));
    for (int i = 0; i < L; ++i)
        CHECK(out[i] == static_cast<float>(i + 1));

    scatter(base, stride, broadcast(-2.0f));
    for (int i = 0; i < L; ++i)
        CHECK(base[i * stride] == -2.0f);

    munmap(region, bytes);
}
#endif
