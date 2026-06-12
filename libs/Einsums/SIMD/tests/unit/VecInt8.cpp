//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Vec<int8_t> / Vec<uint8_t>: round-trip plus the dot-product helpers.
//
// Each dot-product test computes a scalar reference and compares lane-wise.
// Tests gate themselves on the underlying feature flag; `dot_product_ss`
// only runs when FEAT_DOTPROD is on (always true on M-series default
// builds), `dot_product_us` only runs when either FEAT_I8MM (M3+, needs
// `-mcpu=apple-m3` or later) or AVX-VNNI / AVX-512 VNNI is on.

#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Platform.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <cstdint>
#include <vector>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

// ─── Round-trip ────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("8-bit Vec round-trips load/store", "[simd][int8]", int8_t, uint8_t) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> input(N);
    for (int i = 0; i < N; ++i)
        input[i] = static_cast<TestType>(i + 1);
    auto                  v = loadu<TestType>(input.data());
    std::vector<TestType> output(N);
    storeu(output.data(), v);
    for (int i = 0; i < N; ++i)
        CHECK(output[i] == input[i]);
}

TEMPLATE_TEST_CASE("8-bit broadcast fills every lane", "[simd][int8]", int8_t, uint8_t) {
    constexpr int         N = Vec<TestType>::lanes;
    auto                  v = broadcast(static_cast<TestType>(7));
    std::vector<TestType> output(N);
    storeu(output.data(), v);
    for (int i = 0; i < N; ++i)
        CHECK(output[i] == TestType(7));
}

// ─── Dot products ─────────────────────────────────────────────────────────

namespace {
// Reference: scalar 4-byte dot product accumulating into int32. Both
// hardware and the helper aggregate 4 input bytes into one int32 output
// lane. This computes the same per-output-lane scalar sum so we can
// verify lane-wise.
template <typename A, typename B>
std::vector<int32_t> reference_dot4(std::vector<int32_t> const &acc, std::vector<A> const &a, std::vector<B> const &b) {
    int const            out_lanes = static_cast<int>(acc.size());
    std::vector<int32_t> out(out_lanes);
    for (int i = 0; i < out_lanes; ++i) {
        int32_t sum = acc[i];
        for (int k = 0; k < 4; ++k) {
            int32_t const x = static_cast<int32_t>(a[i * 4 + k]);
            int32_t const y = static_cast<int32_t>(b[i * 4 + k]);
            sum += x * y;
        }
        out[i] = sum;
    }
    return out;
}
} // namespace

#if defined(__ARM_FEATURE_DOTPROD)

TEST_CASE("dot_product_ss: signed × signed → int32, accumulating", "[simd][int8][dotprod]") {
    constexpr int byte_lanes  = Vec<int8_t>::lanes;
    constexpr int int32_lanes = Vec<int32_t>::lanes;
    REQUIRE(byte_lanes == int32_lanes * 4);

    std::vector<int8_t>  a(byte_lanes), b(byte_lanes);
    std::vector<int32_t> acc(int32_lanes);
    for (int i = 0; i < byte_lanes; ++i) {
        a[i] = static_cast<int8_t>(i - 8);
        b[i] = static_cast<int8_t>(((i & 1) ? -1 : 1) * (i + 1));
    }
    for (int i = 0; i < int32_lanes; ++i)
        acc[i] = 1000 + i;

    auto va   = loadu<int8_t>(a.data());
    auto vb   = loadu<int8_t>(b.data());
    auto vacc = loadu<int32_t>(acc.data());
    auto vout = dot_product_ss(vacc, va, vb);

    std::vector<int32_t> out(int32_lanes);
    storeu(out.data(), vout);

    auto expected = reference_dot4(acc, a, b);
    for (int i = 0; i < int32_lanes; ++i)
        CHECK(out[i] == expected[i]);
}

TEST_CASE("dot_product_uu: unsigned × unsigned → int32, accumulating", "[simd][int8][dotprod]") {
    constexpr int byte_lanes  = Vec<uint8_t>::lanes;
    constexpr int int32_lanes = Vec<int32_t>::lanes;

    std::vector<uint8_t> a(byte_lanes), b(byte_lanes);
    std::vector<int32_t> acc(int32_lanes);
    for (int i = 0; i < byte_lanes; ++i) {
        a[i] = static_cast<uint8_t>(i + 1);   // 1..N
        b[i] = static_cast<uint8_t>(255 - i); // 255..255-N+1
    }
    for (int i = 0; i < int32_lanes; ++i)
        acc[i] = 0;

    auto va   = loadu<uint8_t>(a.data());
    auto vb   = loadu<uint8_t>(b.data());
    auto vacc = loadu<int32_t>(acc.data());
    auto vout = dot_product_uu(vacc, va, vb);

    std::vector<int32_t> out(int32_lanes);
    storeu(out.data(), vout);

    auto expected = reference_dot4(acc, a, b);
    for (int i = 0; i < int32_lanes; ++i)
        CHECK(out[i] == expected[i]);
}

#endif // FEAT_DOTPROD

#if defined(__ARM_FEATURE_MATMUL_INT8) || defined(__AVX512VNNI__) || defined(__AVXVNNI__)

TEST_CASE("dot_product_us: unsigned × signed → int32, accumulating", "[simd][int8][dotprod]") {
    constexpr int byte_lanes  = Vec<uint8_t>::lanes;
    constexpr int int32_lanes = Vec<int32_t>::lanes;

    std::vector<uint8_t> a(byte_lanes);
    std::vector<int8_t>  b(byte_lanes);
    std::vector<int32_t> acc(int32_lanes);
    for (int i = 0; i < byte_lanes; ++i) {
        a[i] = static_cast<uint8_t>(i + 5);
        b[i] = static_cast<int8_t>(((i % 3) - 1) * (i + 1)); // mix of negatives + zeros
    }
    for (int i = 0; i < int32_lanes; ++i)
        acc[i] = -50 + i * 10;

    auto va   = loadu<uint8_t>(a.data());
    auto vb   = loadu<int8_t>(b.data());
    auto vacc = loadu<int32_t>(acc.data());
    auto vout = dot_product_us(vacc, va, vb);

    std::vector<int32_t> out(int32_lanes);
    storeu(out.data(), vout);

    auto expected = reference_dot4(acc, a, b);
    for (int i = 0; i < int32_lanes; ++i)
        CHECK(out[i] == expected[i]);
}

#endif // I8MM || AVX-VNNI
