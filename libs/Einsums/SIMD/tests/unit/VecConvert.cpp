//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// int32 <-> float conversion: rounding to nearest one way, truncation toward zero the other.

#include <Einsums/SIMD/Convert.hpp>
#include <Einsums/SIMD/Operations.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

TEST_CASE("convert<float> of int32 rounds to nearest, ties to even", "[simd][convert]") {
    constexpr int L = Vec<int32_t>::lanes;
    int32_t       in[L];
    float         out[L];
    for (int i = 0; i < L; ++i) {
        in[i] = (i % 2 ? -1 : 1) * (i * 1000003 + 17);
    }
    storeu(out, convert<float>(loadu(in)));
    for (int i = 0; i < L; ++i) {
        CHECK(out[i] == static_cast<float>(in[i]));
    }
    // 2^24 + 1 and 2^24 + 3 are the first odd integers float cannot hold: the ties go to the even
    // neighbours 2^24 and 2^24 + 4.
    storeu(out, convert<float>(broadcast(int32_t{16777217})));
    CHECK(out[0] == 16777216.0f);
    storeu(out, convert<float>(broadcast(int32_t{16777219})));
    CHECK(out[0] == 16777220.0f);
}

TEST_CASE("convert<int32_t> of float truncates toward zero", "[simd][convert]") {
    constexpr int L = Vec<float>::lanes;
    float         in[L];
    int32_t       out[L];
    for (int i = 0; i < L; ++i) {
        in[i] = (i % 2 ? -1.0f : 1.0f) * (static_cast<float>(i) + 0.75f);
    }
    storeu(out, convert<int32_t>(loadu(in)));
    for (int i = 0; i < L; ++i) {
        INFO("lane " << i << ": " << in[i]);
        CHECK(out[i] == static_cast<int32_t>(in[i]));
    }
    storeu(out, convert<int32_t>(broadcast(-0.5f)));
    CHECK(out[0] == 0);
    storeu(out, convert<int32_t>(broadcast(2147483520.0f))); // the largest float below 2^31
    CHECK(out[0] == 2147483520);
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) ||           \
    defined(__AVX512BF16__)
namespace {

// Widening is exact, so every one of the 65536 bit patterns must come back as the C++ conversion
// gives it, bit for bit; a NaN only has to stay a NaN.
template <typename H>
void check_widen_every_pattern() {
    constexpr int L = Vec<H>::lanes;
    constexpr int F = Vec<float>::lanes;
    static_assert(L == 2 * F);
    for (std::uint32_t base = 0; base < 65536; base += L) {
        H in[L];
        for (int i = 0; i < L; ++i) {
            in[i] = std::bit_cast<H>(static_cast<std::uint16_t>(base + static_cast<std::uint32_t>(i)));
        }
        Vec<H> const v = loadu(in);
        float        low[F], high[F];
        storeu(low, convert_low<float>(v));
        storeu(high, convert_high<float>(v));
        for (int i = 0; i < L; ++i) {
            float const got      = i < F ? low[i] : high[i - F];
            float const expected = static_cast<float>(in[i]);
            if (std::isnan(expected)) {
                CHECK(std::isnan(got));
            } else if (std::bit_cast<std::uint32_t>(got) != std::bit_cast<std::uint32_t>(expected)) {
                FAIL_CHECK("pattern " << (base + i) << ": " << got << " != " << expected);
            }
        }
    }
}

// Narrowing rounds to nearest, ties to even, as the C++ conversion does: checked bit for bit over
// ties, signed zeros, infinities, overflow, subnormal results and a spread of ordinary values.
template <typename H>
void check_narrow(std::vector<float> const &values) {
    constexpr int L = Vec<H>::lanes;
    constexpr int F = Vec<float>::lanes;
    for (std::size_t base = 0; base < values.size(); base += L) {
        float wide[L];
        for (int i = 0; i < L; ++i) {
            wide[i] = values[(base + static_cast<std::size_t>(i)) % values.size()];
        }
        H out[L];
        storeu(out, convert<H>(loadu(wide), loadu(wide + F)));
        for (int i = 0; i < L; ++i) {
            H const expected = static_cast<H>(wide[i]);
            INFO("input " << wide[i]);
            if (std::isnan(wide[i])) {
                CHECK(std::isnan(static_cast<float>(out[i])));
            } else {
                CHECK(std::bit_cast<std::uint16_t>(out[i]) == std::bit_cast<std::uint16_t>(expected));
            }
        }
    }
}

std::vector<float> narrowing_inputs() {
    std::vector<float> v{0.0f, -0.0f, 1.0f, -1.5f, 65504.0f, 65520.0f, 1.0e30f, -1.0e30f, std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::min(), std::numeric_limits<float>::denorm_min(), 6.0e-8f,
                         // bf16 ties: halfway between two bf16 values, so the even one must win.
                         std::bit_cast<float>(0x3F808000u), std::bit_cast<float>(0x3F818000u),
                         // half ties: 1 + 2^-11 and 1 + 3 * 2^-11 sit halfway between half values.
                         1.0f + 0x1p-11f, 1.0f + 3.0f * 0x1p-11f};
    for (int i = 0; i < 64; ++i) {
        v.push_back(static_cast<float>(i * i) * 0.3712f - 413.0f);
    }
    return v;
}

} // namespace
#endif

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
TEST_CASE("half_t widens exactly and narrows to nearest even", "[simd][convert][half]") {
    check_widen_every_pattern<half_t>();
    check_narrow<half_t>(narrowing_inputs());
}
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
TEST_CASE("bfloat16_t widens exactly and narrows to nearest even", "[simd][convert][bf16]") {
    check_widen_every_pattern<bfloat16_t>();
    check_narrow<bfloat16_t>(narrowing_inputs());
}
#endif
