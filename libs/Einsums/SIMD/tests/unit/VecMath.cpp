//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// div, sqrt, min, max, abs and neg against a scalar reference, bit for bit. The inputs pair every
// special value with every other in some lane, so NaN, signed zeros and infinities reach each lane
// position of every backend.

#include <Einsums/SIMD/Operations.hpp>

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
