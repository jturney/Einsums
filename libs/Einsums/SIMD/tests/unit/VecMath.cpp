//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// div, sqrt, min, max, abs, neg, the rounding functions and the fused multiply-add family against a
// scalar reference, bit for bit. The inputs pair every special value with every other in some lane,
// so NaN, signed zeros and infinities reach each lane position of every backend.

#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Platform.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

namespace {

template <typename T>
using bits_t = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;

template <typename T>
constexpr bits_t<T> sign_bit = bits_t<T>{1} << (8 * sizeof(T) - 1);

template <typename T>
std::vector<T> special_values() {
    T const nan = std::numeric_limits<T>::quiet_NaN();
    T const inf = std::numeric_limits<T>::infinity();
    return {T(0), T(-0.0), T(1), T(-1), T(2.5), T(-3.25), nan, inf, -inf, T(1e-30), T(7), T(-7), T(0.5), T(1e30), T(-2), T(3)};
}

/// True when @p got and @p want are the same bits, or both NaN (whose payload no backend promises).
template <typename T>
bool same(T got, T want) {
    return std::bit_cast<bits_t<T>>(got) == std::bit_cast<bits_t<T>>(want) || (std::isnan(got) && std::isnan(want));
}

/// Runs @p op on every pairing of special values and compares each lane with @p ref.
template <typename T, typename Op, typename Ref>
void check_lanes(Op op, Ref ref) {
    constexpr int L = Vec<T>::lanes;
    auto const    s = special_values<T>();
    for (size_t oa = 0; oa < s.size(); ++oa) {
        for (size_t ob = 0; ob < s.size(); ++ob) {
            T a[L], b[L], out[L];
            for (int i = 0; i < L; ++i) {
                a[i] = s[(oa + static_cast<size_t>(i)) % s.size()];
                b[i] = s[(ob + 3 * static_cast<size_t>(i)) % s.size()];
            }
            storeu(out, op(loadu(a), loadu(b)));
            for (int i = 0; i < L; ++i) {
                INFO("lane " << i << ": a = " << a[i] << ", b = " << b[i] << ", got " << out[i]);
                CHECK(same(out[i], ref(a[i], b[i])));
            }
        }
    }
}

} // namespace

TEMPLATE_TEST_CASE("div and operator/ divide lane by lane", "[simd][math]", float, double) {
    check_lanes<TestType>([](auto a, auto b) { return div(a, b); }, [](TestType a, TestType b) { return a / b; });
    check_lanes<TestType>([](auto a, auto b) { return a / b; }, [](TestType a, TestType b) { return a / b; });
}

TEMPLATE_TEST_CASE("sqrt matches std::sqrt, NaN for negatives", "[simd][math]", float, double) {
    check_lanes<TestType>([](auto a, auto) { return sqrt(a); }, [](TestType a, TestType) { return std::sqrt(a); });
}

TEMPLATE_TEST_CASE("min and max are a < b ? a : b and a > b ? a : b exactly", "[simd][math]", float, double) {
    // Including the cases where the backends' own min instructions differ: a NaN on either side,
    // and zeros of opposite sign, all of which must return b.
    check_lanes<TestType>([](auto a, auto b) { return min(a, b); }, [](TestType a, TestType b) { return a < b ? a : b; });
    check_lanes<TestType>([](auto a, auto b) { return max(a, b); }, [](TestType a, TestType b) { return a > b ? a : b; });
}

TEMPLATE_TEST_CASE("abs clears and neg flips only the sign bit", "[simd][math]", float, double) {
    using B = bits_t<TestType>;
    check_lanes<TestType>([](auto a, auto) { return abs(a); },
                          [](TestType a, TestType) { return std::bit_cast<TestType>(std::bit_cast<B>(a) & ~sign_bit<TestType>); });
    check_lanes<TestType>([](auto a, auto) { return neg(a); },
                          [](TestType a, TestType) { return std::bit_cast<TestType>(std::bit_cast<B>(a) ^ sign_bit<TestType>); });
    check_lanes<TestType>([](auto a, auto) { return -a; },
                          [](TestType a, TestType) { return std::bit_cast<TestType>(std::bit_cast<B>(a) ^ sign_bit<TestType>); });
}

namespace {

/// Runs @p op on lanes drawn from @p values, each lane at a different offset so every value reaches
/// every lane position, and compares each lane with @p ref bit for bit.
template <typename T, typename Op, typename Ref>
void check_unary(std::vector<T> const &values, Op op, Ref ref) {
    constexpr int L = Vec<T>::lanes;
    for (size_t o = 0; o < values.size(); ++o) {
        T in[L], out[L];
        for (int i = 0; i < L; ++i) {
            in[i] = values[(o + 5 * static_cast<size_t>(i)) % values.size()];
        }
        storeu(out, op(loadu(in)));
        for (int i = 0; i < L; ++i) {
            INFO("lane " << i << ": x = " << in[i] << ", got " << out[i] << ", want " << ref(in[i]));
            CHECK(same(out[i], ref(in[i])));
        }
    }
}

/// Values that separate the five rounding modes: halves, the largest value below a half, near
/// integers, signed zeros, the magnitudes where the SSE2 emulation stops adding 2^23 or 2^52, and
/// the specials.
template <typename T>
std::vector<T> rounding_values() {
    constexpr T    big = sizeof(T) == 4 ? T(8388608.0) : T(4503599627370496.0); // 2^23, 2^52
    std::vector<T> v   = special_values<T>();
    for (T x : {T(0.5),
                T(1.5),
                T(2.5),
                T(3.5),
                T(0.3),
                T(0.7),
                T(1.0),
                T(2.0),
                T(1e-40),
                T(123456.5),
                big - T(0.5),
                big - T(1.5),
                big,
                big + T(1),
                big + T(2),
                T(2) * big,
                std::nextafter(T(0.5), T(0)),
                std::nextafter(T(1), T(0)),
                std::nextafter(T(1), T(2)),
                std::numeric_limits<T>::max(),
                std::numeric_limits<T>::denorm_min()}) {
        v.push_back(x);
        v.push_back(-x);
    }
    return v;
}

} // namespace

TEMPLATE_TEST_CASE("floor, ceil, trunc, round and round_even match their std functions", "[simd][math][rounding]", float, double) {
    auto const v = rounding_values<TestType>();
    check_unary(v, [](auto x) { return floor(x); }, [](TestType x) { return std::floor(x); });
    check_unary(v, [](auto x) { return ceil(x); }, [](TestType x) { return std::ceil(x); });
    check_unary(v, [](auto x) { return trunc(x); }, [](TestType x) { return std::trunc(x); });
    check_unary(v, [](auto x) { return round(x); }, [](TestType x) { return std::round(x); });
    // std::nearbyint rounds in the current mode, which is to nearest even unless a test changed it.
    check_unary(v, [](auto x) { return round_even(x); }, [](TestType x) { return std::nearbyint(x); });
}

namespace {

/// Whether this build's fmadd family rounds once. On x86 that is has_fma; aarch64 always has FMA;
/// the scalar fallback multiplies and adds separately.
constexpr bool fused_multiply_add =
#if defined(__aarch64__) || defined(_M_ARM64)
    true;
#else
    has_fma;
#endif

/// a * b + c rounded once when the build fuses and twice when it does not. The product goes through
/// a volatile so the compiler cannot contract the unfused reference into an FMA of its own.
template <typename T>
T reference_fma(T a, T b, T c) {
    if constexpr (fused_multiply_add) {
        return std::fma(a, b, c);
    } else {
        T volatile product = a * b;
        return product + c;
    }
}

} // namespace

TEMPLATE_TEST_CASE("fmsub, fnmadd and fnmsub are a*b-c, -(a*b)+c and -(a*b)-c", "[simd][math][fma]", float, double) {
    using T         = TestType;
    constexpr int L = Vec<T>::lanes;
    auto const    s = special_values<T>();
    for (size_t oa = 0; oa < s.size(); ++oa) {
        for (size_t oc = 0; oc < s.size(); ++oc) {
            T a[L], b[L], c[L], ms[L], nma[L], nms[L], ma[L];
            for (int i = 0; i < L; ++i) {
                a[i] = s[(oa + static_cast<size_t>(i)) % s.size()];
                b[i] = s[(oa + oc + 3 * static_cast<size_t>(i)) % s.size()];
                c[i] = s[(oc + 7 * static_cast<size_t>(i)) % s.size()];
            }
            Vec<T> const va = loadu(a), vb = loadu(b), vc = loadu(c);
            storeu(ma, fmadd(va, vb, vc));
            storeu(ms, fmsub(va, vb, vc));
            storeu(nma, fnmadd(va, vb, vc));
            storeu(nms, fnmsub(va, vb, vc));
            for (int i = 0; i < L; ++i) {
                INFO("lane " << i << ": a = " << a[i] << ", b = " << b[i] << ", c = " << c[i]);
                CHECK(same(ma[i], reference_fma(a[i], b[i], c[i])));
                CHECK(same(ms[i], reference_fma(a[i], b[i], -c[i])));
                CHECK(same(nma[i], reference_fma(-a[i], b[i], c[i])));
                CHECK(same(nms[i], reference_fma(-a[i], b[i], -c[i])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("the fmadd family rounds once where the build fuses", "[simd][math][fma]", float, double) {
    // (1 + e)(1 - e) = 1 - e^2, which rounds to exactly 1 when e^2 is below half an ulp of 1. A fused
    // fmsub against c = 1 keeps the -e^2; an unfused one returns zero.
    using T   = TestType;
    T const e = sizeof(T) == 4 ? T(0x1p-13) : T(0x1p-28);
    T       out[Vec<T>::lanes];
    storeu(out, fmsub(broadcast(T(1) + e), broadcast(T(1) - e), broadcast(T(1))));
    CHECK(out[0] == (fused_multiply_add ? -e * e : T(0)));
    storeu(out, fnmadd(broadcast(T(1) + e), broadcast(T(1) - e), broadcast(T(1))));
    CHECK(out[0] == (fused_multiply_add ? e * e : T(0)));
}
