//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Half-precision vector tests.
//
// FP16 (IEEE binary16): full arithmetic surface — broadcast / load / store /
// add / sub / mul / fmadd. Tested whenever the build sees
// `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` (M1+ default) or `__AVX512FP16__`
// (Sapphire Rapids+).
//
// BF16: load/store only — direct arithmetic requires FP32 round-trip.
// Tested when `__ARM_FEATURE_BF16_VECTOR_ARITHMETIC` (M3+, requires
// -mcpu=apple-m3 or later) or `__AVX512BF16__` is set. On the default
// conda toolchain build for Apple Silicon the BF16 tests stay disabled
// until the user adds the appropriate -mcpu flag.

#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Platform.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <cmath>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)

TEST_CASE("FP16 Vec lane count matches register width", "[simd][half]") {
    // 128-bit register on aarch64 → 8 lanes; 512-bit on AVX-512FP16 → 32.
    if constexpr (has_neon_fp16)
        CHECK(Vec<half_t>::lanes == 8);
    else if constexpr (has_avx512_fp16)
        CHECK(Vec<half_t>::lanes == 32);
}

TEST_CASE("FP16 broadcast: every lane holds the input value", "[simd][half]") {
    constexpr int                    N   = Vec<half_t>::lanes;
    half_t const                     val = static_cast<half_t>(2.5);
    auto                             v   = broadcast(val);
    alignas(native_alignment) half_t buf[N];
    storeu(buf, v);
    for (int i = 0; i < N; ++i)
        CHECK(static_cast<float>(buf[i]) == Catch::Approx(2.5f));
}

TEST_CASE("FP16 loadu / storeu round-trip preserves all lanes", "[simd][half]") {
    constexpr int N = Vec<half_t>::lanes;
    half_t        input[N];
    for (int i = 0; i < N; ++i)
        input[i] = static_cast<half_t>(0.5f * float(i + 1));
    auto   v = loadu<half_t>(input);
    half_t output[N];
    storeu(output, v);
    for (int i = 0; i < N; ++i)
        CHECK(static_cast<float>(output[i]) == Catch::Approx(static_cast<float>(input[i])));
}

TEST_CASE("FP16 add: lane-wise sum", "[simd][half]") {
    constexpr int N = Vec<half_t>::lanes;
    half_t        a[N], b[N];
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<half_t>(float(i) + 1.0f);
        b[i] = static_cast<half_t>(0.5f * float(i));
    }
    auto   va = loadu<half_t>(a);
    auto   vb = loadu<half_t>(b);
    auto   vc = add(va, vb);
    half_t out[N];
    storeu(out, vc);
    for (int i = 0; i < N; ++i)
        CHECK(static_cast<float>(out[i]) == Catch::Approx(static_cast<float>(a[i]) + static_cast<float>(b[i])).epsilon(1e-3));
}

TEST_CASE("FP16 sub / mul lane-wise behavior", "[simd][half]") {
    constexpr int N = Vec<half_t>::lanes;
    half_t        a[N], b[N];
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<half_t>(float(i) + 2.0f);
        b[i] = static_cast<half_t>(float(i) + 1.0f);
    }
    auto   va = loadu<half_t>(a);
    auto   vb = loadu<half_t>(b);
    half_t sub_out[N], mul_out[N];
    storeu(sub_out, sub(va, vb));
    storeu(mul_out, mul(va, vb));
    for (int i = 0; i < N; ++i) {
        CHECK(static_cast<float>(sub_out[i]) == Catch::Approx(static_cast<float>(a[i]) - static_cast<float>(b[i])).epsilon(1e-3));
        CHECK(static_cast<float>(mul_out[i]) == Catch::Approx(static_cast<float>(a[i]) * static_cast<float>(b[i])).epsilon(1e-3));
    }
}

TEST_CASE("FP16 fmadd computes a*b + c", "[simd][half]") {
    constexpr int N = Vec<half_t>::lanes;
    half_t        a[N], b[N], c[N];
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<half_t>(2.0f);
        b[i] = static_cast<half_t>(0.5f * float(i));
        c[i] = static_cast<half_t>(1.0f);
    }
    auto   va = loadu<half_t>(a);
    auto   vb = loadu<half_t>(b);
    auto   vc = loadu<half_t>(c);
    auto   vr = fmadd(va, vb, vc);
    half_t out[N];
    storeu(out, vr);
    for (int i = 0; i < N; ++i) {
        float const expected = static_cast<float>(a[i]) * static_cast<float>(b[i]) + static_cast<float>(c[i]);
        CHECK(static_cast<float>(out[i]) == Catch::Approx(expected).epsilon(1e-3));
    }
}

#endif // FP16

// ─── BF16 ─────────────────────────────────────────────────────────────────

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)

TEST_CASE("BF16 Vec lane count matches register width", "[simd][bf16]") {
    if constexpr (has_neon_bf16)
        CHECK(Vec<bfloat16_t>::lanes == 8);
    else if constexpr (has_avx512_bf16)
        CHECK(Vec<bfloat16_t>::lanes == 32);
}

TEST_CASE("BF16 loadu / storeu round-trip preserves bit pattern", "[simd][bf16]") {
    // BF16 has no native scalar arithmetic on the type, so the simplest
    // way to construct test values is via uint16_t bit patterns. The
    // storage round-trip must preserve those exactly.
    constexpr int                      N = Vec<bfloat16_t>::lanes;
    alignas(native_alignment) uint16_t pattern[N];
    for (int i = 0; i < N; ++i)
        pattern[i] = static_cast<uint16_t>(0x3F80 + i); // ~1.0f, 1.0f+ε, ...
    auto                              *as_bf = reinterpret_cast<bfloat16_t *>(pattern);
    auto                               v     = loadu<bfloat16_t>(as_bf);
    alignas(native_alignment) uint16_t output[N];
    storeu(reinterpret_cast<bfloat16_t *>(output), v);
    for (int i = 0; i < N; ++i)
        CHECK(output[i] == pattern[i]);
}

#endif // BF16
