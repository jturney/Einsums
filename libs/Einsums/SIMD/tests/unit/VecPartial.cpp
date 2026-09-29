//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Partial loads and stores at every tail length. The exact-size cases allocate exactly n elements
// on the heap, so an access past them is an out-of-bounds read or write that the sanitizer legs
// report, rather than a quiet read of whatever sits next in a larger buffer.

#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Partial.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

TEMPLATE_TEST_CASE("loadu_partial reads n lanes and zeroes the rest", "[simd][partial]", float, double, int32_t, uint32_t, int64_t,
                   uint64_t) {
    constexpr int         L = Vec<TestType>::lanes;
    std::vector<TestType> src(L);
    for (int i = 0; i < L; ++i) {
        src[i] = TestType(i) + TestType(1);
    }
    for (std::size_t n = 0; n <= static_cast<std::size_t>(L) + 1; ++n) {
        TestType out[L];
        storeu(out, loadu_partial(src.data(), n));
        for (int i = 0; i < L; ++i) {
            INFO("n = " << n << ", lane " << i);
            CHECK(out[i] == (static_cast<std::size_t>(i) < n ? src[i] : TestType(0)));
        }
    }
}

TEMPLATE_TEST_CASE("storeu_partial writes n lanes and nothing past them", "[simd][partial]", float, double, int32_t, uint32_t, int64_t,
                   uint64_t) {
    constexpr int L = Vec<TestType>::lanes;
    TestType      src[L];
    for (int i = 0; i < L; ++i) {
        src[i] = TestType(i) + TestType(1);
    }
    for (std::size_t n = 0; n <= static_cast<std::size_t>(L) + 1; ++n) {
        std::vector<TestType> dst(L + 2, TestType(99));
        storeu_partial(dst.data(), loadu(src), n);
        for (int i = 0; i < L + 2; ++i) {
            INFO("n = " << n << ", element " << i);
            CHECK(dst[i] == (static_cast<std::size_t>(i) < std::min(n, static_cast<std::size_t>(L)) ? src[i] : TestType(99)));
        }
    }
}

TEMPLATE_TEST_CASE("partial load and store stay inside an exact-size allocation", "[simd][partial]", float, double, int32_t, int64_t) {
    constexpr int L = Vec<TestType>::lanes;
    for (std::size_t n = 1; n < static_cast<std::size_t>(L); ++n) {
        auto const src = std::make_unique<TestType[]>(n);
        auto const dst = std::make_unique<TestType[]>(n);
        for (std::size_t i = 0; i < n; ++i) {
            src[i] = TestType(i) + TestType(1);
        }
        storeu_partial(dst.get(), loadu_partial(src.get(), n), n);
        for (std::size_t i = 0; i < n; ++i) {
            CHECK(dst[i] == src[i]);
        }
    }
}

TEST_CASE("native_partial is true exactly where partial access is a masked instruction", "[simd][partial]") {
#if defined(__AVX512F__) && defined(__AVX512VL__)
    STATIC_CHECK(native_partial<float>);
    STATIC_CHECK(native_partial<double>);
    STATIC_CHECK(native_partial<int32_t>);
    STATIC_CHECK(native_partial<int64_t>);
#elif defined(__AVX__)
    STATIC_CHECK(native_partial<float>);
    STATIC_CHECK(native_partial<double>);
    STATIC_CHECK_FALSE(native_partial<int32_t>);
#else
    STATIC_CHECK_FALSE(native_partial<float>);
    STATIC_CHECK_FALSE(native_partial<double>);
    STATIC_CHECK_FALSE(native_partial<int32_t>);
#endif
}
