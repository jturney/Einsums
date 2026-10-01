//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Mask<T>: the comparisons, select, the mask logic and operators, any/all/none/count, first_n, the
// bit conversions and mask_cast, against scalar references. to_vec must give exactly all-ones or
// zero per lane, so masks are compared bit for bit.

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

/// Checks that @p mask holds exactly ref(a[i], b[i]) in every lane, both as bits (to_vec gives
/// all-ones or zero per lane) and as the integer to_bits gives.
template <typename T, typename Ref>
void check_mask(Mask<T> mask, T const *a, T const *b, Ref ref) {
    constexpr int L = Vec<T>::lanes;
    T             out[L];
    storeu(out, to_vec(mask));
    uint64_t const bits = to_bits(mask);
    for (int i = 0; i < L; ++i) {
        INFO("lane " << i << ": a = " << a[i] << ", b = " << b[i]);
        CHECK(std::bit_cast<bits_t<T>>(out[i]) == std::bit_cast<bits_t<T>>(mask_lane<T>(ref(a[i], b[i]))));
        CHECK(((bits >> i) & 1u) == (ref(a[i], b[i]) ? 1u : 0u));
    }
    CHECK((bits >> L) == 0u);
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
        check_mask(lt & le, a, b, [](TestType x, TestType y) { return x < y && x <= y; });
        check_mask(lt | gt, a, b, [](TestType x, TestType y) { return x < y || x > y; });
        check_mask(lt ^ le, a, b, [](TestType x, TestType y) { return (x < y) != (x <= y); });
        check_mask(!lt, a, b, [](TestType x, TestType y) { return !(x < y); });
        auto compound = lt;
        compound |= gt;
        compound &= le;
        compound ^= lt;
        check_mask(compound, a, b, [](TestType x, TestType y) { return (((x < y) || (x > y)) && (x <= y)) != (x < y); });
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

TEMPLATE_TEST_CASE("first_n sets the first n lanes, and the bit conversions round-trip", "[simd][mask]", float, double, int32_t, uint32_t,
                   int64_t, uint64_t) {
    using T         = TestType;
    constexpr int L = Vec<T>::lanes;
    for (std::size_t n = 0; n <= static_cast<std::size_t>(L) + 2; ++n) {
        Mask<T> const     m    = first_n<T>(n);
        std::size_t const set  = n < static_cast<std::size_t>(L) ? n : static_cast<std::size_t>(L);
        uint64_t const    want = (uint64_t{1} << set) - 1u;
        INFO("n = " << n);
        CHECK(to_bits(m) == want);
        CHECK(count(m) == static_cast<int>(set));
        CHECK(any(m) == (set > 0));
        CHECK(none(m) == (set == 0));
        CHECK(all(m) == (set == static_cast<std::size_t>(L)));
        CHECK(to_bits(!m) == ((uint64_t{1} << L) - 1u - want));
    }
    CHECK(to_bits(mask_all<T>()) == (uint64_t{1} << L) - 1u);
    CHECK(to_bits(mask_none<T>()) == 0u);
    for (uint64_t bits = 0; bits < (uint64_t{1} << L) && bits < 4096; bits = bits * 3 + 1) {
        CHECK(to_bits(mask_from_bits<T>(bits)) == bits);
        CHECK(count(mask_from_bits<T>(bits)) == std::popcount(bits));
    }
    // Bits past the lane count are ignored.
    CHECK(to_bits(mask_from_bits<T>(~uint64_t{0})) == (uint64_t{1} << L) - 1u);
}

TEST_CASE("mask_cast keeps the lanes between element types of one width", "[simd][mask]") {
    for (std::size_t n = 0; n <= static_cast<std::size_t>(Vec<float>::lanes); ++n) {
        CHECK(to_bits(mask_cast<int32_t>(first_n<float>(n))) == to_bits(first_n<float>(n)));
        CHECK(to_bits(mask_cast<float>(first_n<uint32_t>(n))) == to_bits(first_n<uint32_t>(n)));
    }
    for (std::size_t n = 0; n <= static_cast<std::size_t>(Vec<double>::lanes); ++n) {
        CHECK(to_bits(mask_cast<int64_t>(first_n<double>(n))) == to_bits(first_n<double>(n)));
        CHECK(to_bits(mask_cast<double>(first_n<uint64_t>(n))) == to_bits(first_n<uint64_t>(n)));
    }
}

TEMPLATE_TEST_CASE("a masked fmadd is select over fmadd", "[simd][mask]", float, double) {
    // Active lanes take fmadd's result exactly, inactive lanes keep c.
    using T = TestType;
    for_each_pairing<T>([](T const *a, T const *b) {
        constexpr int L  = Vec<T>::lanes;
        Vec<T> const  va = loadu(a), vb = loadu(b);
        T             out[L], fused[L];
        storeu(out, select(cmp_lt(va, vb), fmadd(va, vb, vb), vb));
        storeu(fused, fmadd(va, vb, vb));
        for (int i = 0; i < L; ++i) {
            T const want = a[i] < b[i] ? fused[i] : b[i];
            INFO("lane " << i << ": a = " << a[i] << ", b = " << b[i]);
            CHECK(std::bit_cast<bits_t<T>>(out[i]) == std::bit_cast<bits_t<T>>(want));
        }
    });
}
