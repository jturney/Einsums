//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Debugging.hpp>
#include <Einsums/SIMD/Gather.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Platform.hpp>
#include <Einsums/SIMD/Prefetch.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

TEST_CASE("Platform detection", "[simd]") {
    // At least one platform should be detected (or scalar fallback)
    INFO("native_bits      = " << native_bits);
    INFO("has_sse2         = " << has_sse2);
    INFO("has_avx          = " << has_avx);
    INFO("has_avx2         = " << has_avx2);
    INFO("has_avx512       = " << has_avx512);
    INFO("has_avx10_1      = " << has_avx10_1);
    INFO("has_avx10_2      = " << has_avx10_2);
    INFO("has_avx10_256    = " << has_avx10_256);
    INFO("has_avx10_512    = " << has_avx10_512);
    INFO("has_avx512_fp16  = " << has_avx512_fp16);
    INFO("has_avx512_bf16  = " << has_avx512_bf16);
    INFO("has_neon         = " << has_neon);
    INFO("has_neon_fp16    = " << has_neon_fp16);
    INFO("has_neon_bf16    = " << has_neon_bf16);
    INFO("has_neon_i8mm    = " << has_neon_i8mm);

    // native_bits must be one of the known values
    CHECK((native_bits == 0 || native_bits == 128 || native_bits == 256 || native_bits == 512));

    // native_lanes must be consistent
    CHECK(native_lanes<float> == native_bits / 32);
    CHECK(native_lanes<double> == native_bits / 64);

    // AVX-10 invariants:
    //   - AVX-10/256 chips have AVX2 but NOT __AVX512F__ (so has_avx512 is false).
    //   - AVX-10/512 chips have full AVX-512, so has_avx512 implies has_avx10_512
    //     OR the chip predates AVX-10 entirely (Skylake-X through Ice Lake).
    //   - AVX-10/v2 implies AVX-10/v1 (always-additive versioning).
    if (has_avx10_2)
        CHECK(has_avx10_1);
    if (has_avx10_512)
        CHECK(has_avx10_1);
    // AVX-10/256-only chips: 256-bit max width, AVX2 active, AVX-512 inactive.
    if (has_avx10_256 && !has_avx10_512)
        CHECK_FALSE(has_avx512);
}

TEMPLATE_TEST_CASE("Vec lanes match native_lanes", "[simd]", float, double) {
    CHECK(Vec<TestType>::lanes == native_lanes<TestType>);
}

TEMPLATE_TEST_CASE("broadcast and element access", "[simd]", float, double) {
    constexpr int N = Vec<TestType>::lanes;
    auto          v = broadcast(TestType(3.14));
    for (int i = 0; i < N; ++i) {
        CHECK(v[i] == Catch::Approx(TestType(3.14)));
    }
}

TEMPLATE_TEST_CASE("loadu / storeu round-trip", "[simd]", float, double) {
    constexpr int                      N = Vec<TestType>::lanes;
    alignas(native_alignment) TestType src[N];
    alignas(native_alignment) TestType dst[N];

    for (int i = 0; i < N; ++i)
        src[i] = TestType(i + 1);

    auto v = loadu(src);
    storeu(dst, v);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i] == Catch::Approx(src[i]));
    }
}

TEMPLATE_TEST_CASE("loada / storea round-trip", "[simd]", float, double) {
    constexpr int                      N = Vec<TestType>::lanes;
    alignas(native_alignment) TestType src[N];
    alignas(native_alignment) TestType dst[N];

    for (int i = 0; i < N; ++i)
        src[i] = TestType(i * 2 + 0.5);

    auto v = loada(src);
    storea(dst, v);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i] == Catch::Approx(src[i]));
    }
}

TEMPLATE_TEST_CASE("add", "[simd]", float, double) {
    constexpr int N = Vec<TestType>::lanes;
    auto          a = broadcast(TestType(2.0));
    auto          b = broadcast(TestType(3.0));
    auto          c = add(a, b);

    for (int i = 0; i < N; ++i) {
        CHECK(c[i] == Catch::Approx(TestType(5.0)));
    }
}

TEMPLATE_TEST_CASE("sub", "[simd]", float, double) {
    constexpr int N = Vec<TestType>::lanes;
    auto          a = broadcast(TestType(7.0));
    auto          b = broadcast(TestType(3.0));
    auto          c = sub(a, b);

    for (int i = 0; i < N; ++i) {
        CHECK(c[i] == Catch::Approx(TestType(4.0)));
    }
}

TEMPLATE_TEST_CASE("mul", "[simd]", float, double) {
    constexpr int N = Vec<TestType>::lanes;
    auto          a = broadcast(TestType(4.0));
    auto          b = broadcast(TestType(2.5));
    auto          c = mul(a, b);

    for (int i = 0; i < N; ++i) {
        CHECK(c[i] == Catch::Approx(TestType(10.0)));
    }
}

TEMPLATE_TEST_CASE("fmadd: a*b + c", "[simd]", float, double) {
    constexpr int N = Vec<TestType>::lanes;
    auto          a = broadcast(TestType(2.0));
    auto          b = broadcast(TestType(3.0));
    auto          c = broadcast(TestType(1.0));
    auto          r = fmadd(a, b, c);

    for (int i = 0; i < N; ++i) {
        CHECK(r[i] == Catch::Approx(TestType(7.0))); // 2*3 + 1
    }
}

TEMPLATE_TEST_CASE("operator overloads", "[simd]", float, double) {
    constexpr int N = Vec<TestType>::lanes;
    auto          a = broadcast(TestType(2.0));
    auto          b = broadcast(TestType(3.0));

    auto sum  = a + b;
    auto diff = a - b;
    auto prod = a * b;

    for (int i = 0; i < N; ++i) {
        CHECK(sum[i] == Catch::Approx(TestType(5.0)));
        CHECK(diff[i] == Catch::Approx(TestType(-1.0)));
        CHECK(prod[i] == Catch::Approx(TestType(6.0)));
    }
}

TEMPLATE_TEST_CASE("stream_store", "[simd]", float, double) {
    constexpr int                      N = Vec<TestType>::lanes;
    alignas(native_alignment) TestType dst[N];

    auto v = broadcast(TestType(42.0));
    stream_store(dst, v);

    for (int i = 0; i < N; ++i) {
        CHECK(dst[i] == Catch::Approx(TestType(42.0)));
    }
}
