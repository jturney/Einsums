//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Conversions between element types: int32 and float, int64 and double, float and double, int32 and
// double, the 16-bit floats, and bitcast. Integer to float rounds to nearest, float to integer
// truncates toward zero, and widening is exact.

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

// ─── bitcast ──────────────────────────────────────────────────────────────

TEST_CASE("bitcast reinterprets the bits of each lane", "[simd][convert][bitcast]") {
    constexpr int F = Vec<float>::lanes;
    constexpr int D = Vec<double>::lanes;
    float         f[F];
    double        d[D];
    for (int i = 0; i < F; ++i) {
        f[i] = (i % 2 ? -1.0f : 1.0f) * (static_cast<float>(i) + 0.25f);
    }
    for (int i = 0; i < D; ++i) {
        d[i] = (i % 2 ? -1.0 : 1.0) * (static_cast<double>(i) + 0.125);
    }
    int32_t  fi[F];
    uint32_t fu[F];
    int64_t  di[D];
    uint64_t du[D];
    storeu(fi, bitcast<int32_t>(loadu(f)));
    storeu(fu, bitcast<uint32_t>(loadu(f)));
    storeu(di, bitcast<int64_t>(loadu(d)));
    storeu(du, bitcast<uint64_t>(loadu(d)));
    for (int i = 0; i < F; ++i) {
        CHECK(fi[i] == std::bit_cast<int32_t>(f[i]));
        CHECK(fu[i] == std::bit_cast<uint32_t>(f[i]));
    }
    for (int i = 0; i < D; ++i) {
        CHECK(di[i] == std::bit_cast<int64_t>(d[i]));
        CHECK(du[i] == std::bit_cast<uint64_t>(d[i]));
    }
    // And back, through both signednesses.
    float  f2[F];
    double d2[D];
    storeu(f2, bitcast<float>(bitcast<uint32_t>(bitcast<int32_t>(loadu(f)))));
    storeu(d2, bitcast<double>(bitcast<uint64_t>(bitcast<int64_t>(loadu(d)))));
    for (int i = 0; i < F; ++i) {
        CHECK(std::bit_cast<uint32_t>(f2[i]) == std::bit_cast<uint32_t>(f[i]));
    }
    for (int i = 0; i < D; ++i) {
        CHECK(std::bit_cast<uint64_t>(d2[i]) == std::bit_cast<uint64_t>(d[i]));
    }
}

// ─── double and int64 ─────────────────────────────────────────────────────

namespace {

/// Every value lands in every lane, next to every other, so a batch can mix lanes that take the
/// x86 fast path (below 2^51) with lanes that force the lane-by-lane one.
template <typename From, typename To, typename Ref>
void check_same_lane_conversion(std::vector<From> const &values, Ref ref) {
    constexpr int L = Vec<From>::lanes;
    for (size_t o = 0; o < values.size(); ++o) {
        for (size_t step : {size_t{1}, size_t{3}}) {
            From in[L];
            To   out[L];
            for (int i = 0; i < L; ++i) {
                in[i] = values[(o + step * static_cast<size_t>(i)) % values.size()];
            }
            storeu(out, convert<To>(loadu(in)));
            for (int i = 0; i < L; ++i) {
                INFO("lane " << i << ": " << in[i]);
                ref(in[i], out[i]);
            }
        }
    }
}

} // namespace

TEST_CASE("convert<int64_t> of double truncates toward zero", "[simd][convert]") {
    constexpr double    two51 = 2251799813685248.0, two63 = 9223372036854775808.0;
    std::vector<double> in{0.0,
                           -0.0,
                           0.75,
                           -0.75,
                           1.5,
                           -2.5,
                           123456789.875,
                           -987654321.25,
                           two51,
                           -two51,
                           two51 - 0.5,
                           -two51 + 0.5,
                           two51 + 1,
                           4503599627370497.0,
                           1e18,
                           -1e18,
                           -two63,
                           std::nextafter(two63, 0.0),
                           -std::nextafter(two63, 0.0)};
    check_same_lane_conversion<double, int64_t>(in, [](double x, int64_t got) { CHECK(got == static_cast<int64_t>(x)); });

    // Out of range or NaN: x86 gives INT64_MIN and NEON saturates; either is allowed.
    std::vector<double> bad{two63, -two63 * 2, 1e300, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
                            0.5};
    check_same_lane_conversion<double, int64_t>(bad, [](double x, int64_t got) {
        if (x == 0.5) {
            CHECK(got == 0);
        } else {
            CHECK((got == std::numeric_limits<int64_t>::min() || got == std::numeric_limits<int64_t>::max() || got == 0));
        }
    });
}

TEST_CASE("convert<double> of int64 rounds to nearest, ties to even", "[simd][convert]") {
    constexpr int64_t    two51 = int64_t{1} << 51, two53 = int64_t{1} << 53;
    std::vector<int64_t> in{0,
                            1,
                            -1,
                            42,
                            -123456789,
                            two51 - 1,
                            -(two51 - 1),
                            two51,
                            -two51,
                            two53,
                            two53 + 1,
                            two53 + 3,
                            -(two53 + 1),
                            -(two53 + 3),
                            std::numeric_limits<int64_t>::max(),
                            std::numeric_limits<int64_t>::min()};
    check_same_lane_conversion<int64_t, double>(in, [](int64_t x, double got) { CHECK(got == static_cast<double>(x)); });
}

// ─── float and double, double and int32 ───────────────────────────────────

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__aarch64__) || defined(_M_ARM64)
TEST_CASE("float widens to double exactly and narrows to nearest even", "[simd][convert]") {
    constexpr int F = Vec<float>::lanes;
    constexpr int D = Vec<double>::lanes;
    static_assert(F == 2 * D);
    float in[F];
    for (int i = 0; i < F; ++i) {
        in[i] = (i % 2 ? -1.0f : 1.0f) * std::ldexp(1.0f + static_cast<float>(i) / 16.0f, 3 * i - 20);
    }
    in[1] = std::numeric_limits<float>::denorm_min();
    in[2] = std::numeric_limits<float>::infinity();
    double low[D], high[D];
    storeu(low, convert_low<double>(loadu(in)));
    storeu(high, convert_high<double>(loadu(in)));
    for (int i = 0; i < F; ++i) {
        CHECK((i < D ? low[i] : high[i - D]) == static_cast<double>(in[i]));
    }

    // Narrowing: a tie between two floats goes to the even one, a value past FLT_MAX overflows to
    // infinity, one below the smallest subnormal rounds to a signed zero, and NaN stays NaN.
    std::vector<double> wide{
        1.0 + 0x1p-24, 1.0 + 3 * 0x1p-24, -(1.0 + 0x1p-24), 3.5e38, -3.5e38, 1e-50, -1e-50, std::numeric_limits<double>::quiet_NaN(), 0.1,
        -0.0,          1.0 / 3.0,         16777217.0};
    for (size_t o = 0; o < wide.size(); ++o) {
        double lo[D], hi[D];
        for (int i = 0; i < D; ++i) {
            lo[i] = wide[(o + static_cast<size_t>(i)) % wide.size()];
            hi[i] = wide[(o + static_cast<size_t>(D + i)) % wide.size()];
        }
        float out[F];
        storeu(out, convert<float>(loadu(lo), loadu(hi)));
        for (int i = 0; i < F; ++i) {
            double const x = i < D ? lo[i] : hi[i - D];
            INFO("input " << x);
            if (std::isnan(x)) {
                CHECK(std::isnan(out[i]));
            } else {
                CHECK(std::bit_cast<uint32_t>(out[i]) == std::bit_cast<uint32_t>(static_cast<float>(x)));
            }
        }
    }
}

TEST_CASE("int32 widens to double exactly and double truncates into int32", "[simd][convert]") {
    constexpr int I = Vec<int32_t>::lanes;
    constexpr int D = Vec<double>::lanes;
    static_assert(I == 2 * D);
    int32_t in[I];
    for (int i = 0; i < I; ++i) {
        in[i] = (i % 2 ? -1 : 1) * (i * 134217727 + 7);
    }
    in[0]     = std::numeric_limits<int32_t>::min();
    in[I - 1] = std::numeric_limits<int32_t>::max();
    double low[D], high[D];
    storeu(low, convert_low<double>(loadu(in)));
    storeu(high, convert_high<double>(loadu(in)));
    for (int i = 0; i < I; ++i) {
        CHECK((i < D ? low[i] : high[i - D]) == static_cast<double>(in[i]));
    }

    double lo[D], hi[D];
    for (int i = 0; i < D; ++i) {
        lo[i] = (i % 2 ? -1.0 : 1.0) * (static_cast<double>(i) * 1000.5 + 0.75);
        hi[i] = (i % 2 ? 1.0 : -1.0) * (2147483647.75 - static_cast<double>(i));
    }
    int32_t out[I];
    storeu(out, convert<int32_t>(loadu(lo), loadu(hi)));
    for (int i = 0; i < I; ++i) {
        double const x = i < D ? lo[i] : hi[i - D];
        INFO("input " << x);
        CHECK(out[i] == static_cast<int32_t>(x));
    }
}
#else
TEST_CASE("the scalar build converts float, double and int32 lane for lane", "[simd][convert]") {
    CHECK(convert<double>(broadcast(1.5f))[0] == 1.5);
    CHECK(convert<float>(broadcast(1.0 + 0x1p-24))[0] == 1.0f);
    CHECK(convert<double>(broadcast(int32_t{-7}))[0] == -7.0);
    CHECK(convert<int32_t>(broadcast(-7.75))[0] == -7);
}
#endif
