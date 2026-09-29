//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Floating-point comparisons, select, mask logic and any/all, against scalar references. A mask
// lane must be exactly all-ones or zero, so masks are compared bit for bit.

#include <Einsums/SIMD/Operations.hpp>

#include <bit>
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
T mask_lane(bool set) {
    return std::bit_cast<T>(set ? ~bits_t<T>{0} : bits_t<T>{0});
}

template <typename T>
std::vector<T> special_values() {
    T const nan = std::numeric_limits<T>::quiet_NaN();
    T const inf = std::numeric_limits<T>::infinity();
    return {T(0), T(-0.0), T(1), T(-1), T(2.5), nan, inf, -inf, T(7), T(-7), T(0.5), T(3)};
}

/// Visits every pairing of special values, one pair per lane position.
template <typename T, typename F>
void for_each_pairing(F f) {
    constexpr int L = Vec<T>::lanes;
    auto const    s = special_values<T>();
    for (size_t oa = 0; oa < s.size(); ++oa) {
        for (size_t ob = 0; ob < s.size(); ++ob) {
            T a[L], b[L];
            for (int i = 0; i < L; ++i) {
                a[i] = s[(oa + static_cast<size_t>(i)) % s.size()];
                b[i] = s[(ob + 5 * static_cast<size_t>(i)) % s.size()];
            }
            f(a, b);
        }
    }
}

/// Checks that @p mask holds exactly ref(a[i], b[i]) in every lane.
template <typename T, typename Ref>
void check_mask(Vec<T> mask, T const *a, T const *b, Ref ref) {
    constexpr int L = Vec<T>::lanes;
    T             out[L];
    storeu(out, mask);
    for (int i = 0; i < L; ++i) {
        INFO("lane " << i << ": a = " << a[i] << ", b = " << b[i]);
        CHECK(std::bit_cast<bits_t<T>>(out[i]) == std::bit_cast<bits_t<T>>(mask_lane<T>(ref(a[i], b[i]))));
    }
}

} // namespace

TEMPLATE_TEST_CASE("comparisons follow IEEE 754, NaN true only for cmp_ne", "[simd][mask]", float, double) {
    for_each_pairing<TestType>([](TestType const *a, TestType const *b) {
        auto const va = loadu(a), vb = loadu(b);
        check_mask(cmp_eq(va, vb), a, b, [](TestType x, TestType y) { return x == y; });
        check_mask(cmp_ne(va, vb), a, b, [](TestType x, TestType y) { return x != y; });
        check_mask(cmp_lt(va, vb), a, b, [](TestType x, TestType y) { return x < y; });
        check_mask(cmp_le(va, vb), a, b, [](TestType x, TestType y) { return x <= y; });
        check_mask(cmp_gt(va, vb), a, b, [](TestType x, TestType y) { return x > y; });
        check_mask(cmp_ge(va, vb), a, b, [](TestType x, TestType y) { return x >= y; });
    });
}

TEMPLATE_TEST_CASE("mask logic combines comparison results", "[simd][mask]", float, double) {
    for_each_pairing<TestType>([](TestType const *a, TestType const *b) {
        auto const va = loadu(a), vb = loadu(b);
        auto const lt = cmp_lt(va, vb), gt = cmp_gt(va, vb), le = cmp_le(va, vb);
        check_mask(bitwise_and(lt, le), a, b, [](TestType x, TestType y) { return x < y && x <= y; });
        check_mask(bitwise_or(lt, gt), a, b, [](TestType x, TestType y) { return x < y || x > y; });
        check_mask(bitwise_xor(lt, le), a, b, [](TestType x, TestType y) { return (x < y) != (x <= y); });
        check_mask(bitwise_andnot(le, lt), a, b, [](TestType x, TestType y) { return x <= y && !(x < y); });
    });
}

TEMPLATE_TEST_CASE("select takes a where the mask is set and b elsewhere", "[simd][mask]", float, double) {
    constexpr int L = Vec<TestType>::lanes;
    for_each_pairing<TestType>([](TestType const *a, TestType const *b) {
        TestType out[L];
        storeu(out, select(cmp_lt(loadu(a), loadu(b)), loadu(a), loadu(b)));
        for (int i = 0; i < L; ++i) {
            TestType const want = a[i] < b[i] ? a[i] : b[i];
            INFO("lane " << i);
            CHECK(std::bit_cast<bits_t<TestType>>(out[i]) == std::bit_cast<bits_t<TestType>>(want));
        }
    });
}

TEMPLATE_TEST_CASE("any and all read every lane of a mask", "[simd][mask]", float, double) {
    constexpr int L = Vec<TestType>::lanes;
    for_each_pairing<TestType>([](TestType const *a, TestType const *b) {
        bool any_ref = false, all_ref = true;
        for (int i = 0; i < L; ++i) {
            any_ref = any_ref || a[i] < b[i];
            all_ref = all_ref && a[i] < b[i];
        }
        auto const lt = cmp_lt(loadu(a), loadu(b));
        CHECK(any(lt) == any_ref);
        CHECK(all(lt) == all_ref);
    });
    // One set lane at each position, so a backend that reads only some lanes fails here.
    for (int set = 0; set < L; ++set) {
        TestType v[L];
        for (int i = 0; i < L; ++i) {
            v[i] = i == set ? TestType(1) : TestType(0);
        }
        auto const m = cmp_gt(loadu(v), broadcast(TestType(0)));
        CHECK(any(m));
        CHECK(all(m) == (L == 1));
    }
}

TEMPLATE_TEST_CASE("select, andnot, any and all on integer masks", "[simd][mask][integer]", int32_t, uint32_t, int64_t, uint64_t) {
    constexpr int L = Vec<TestType>::lanes;
    TestType      a[L], b[L], out[L];
    for (int i = 0; i < L; ++i) {
        a[i] = TestType(i);
        b[i] = TestType(i % 2 ? i : 100 + i);
    }
    auto const eq = cmp_eq(loadu(a), loadu(b));
    storeu(out, select(eq, broadcast(TestType(1)), broadcast(TestType(2))));
    for (int i = 0; i < L; ++i) {
        CHECK(out[i] == (a[i] == b[i] ? TestType(1) : TestType(2)));
    }
    storeu(out, bitwise_andnot(broadcast(TestType(0xF0)), broadcast(TestType(0x30))));
    for (int i = 0; i < L; ++i) {
        CHECK(out[i] == TestType(0xC0));
    }
    CHECK(any(eq) == (L > 1));
    CHECK(all(eq) == (L == 1 && a[0] == b[0]));
    CHECK(all(cmp_eq(loadu(a), loadu(a))));
    CHECK_FALSE(any(cmp_eq(loadu(a), broadcast(TestType(200)))));
}
