//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// One kernel body for vectors and scalars (Generic.hpp). Each operation is written once, as a generic
// lambda, and run twice: on a Vec<T>, and on each lane's inputs as plain T. Every lane of the vector
// run must equal the scalar run bit for bit, so a scalar instantiation is an exact reference for the
// vector one, over NaN, signed zeros, infinities and the values that separate the rounding modes.

#include <Einsums/SIMD/Generic.hpp>
#include <Einsums/SIMD/Math.hpp>
#include <Einsums/SIMD/RungLadder.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <catch2/catch_all.hpp>

namespace simd = einsums::simd;

namespace {

template <typename T>
using bits_t = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;

template <typename T>
bool same(T got, T want) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::bit_cast<bits_t<T>>(got) == std::bit_cast<bits_t<T>>(want) || (std::isnan(got) && std::isnan(want));
    } else {
        return got == want;
    }
}

template <typename T>
std::vector<T> inputs() {
    T const nan = std::numeric_limits<T>::quiet_NaN();
    T const inf = std::numeric_limits<T>::infinity();
    return {T(0),       T(-0.0), T(1),   T(-1),   T(2.5),  T(-3.25), nan,     inf,     -inf,  T(1e-30), T(7),
            T(-7),      T(0.5),  T(1.5), T(-1.5), T(-0.5), T(0.3),   T(-0.7), T(1e30), T(-2), T(3),     std::nextafter(T(0.5), T(0)),
            T(123456.5)};
}

/// Run op on a V whose lanes are drawn from the inputs, and on each lane's inputs as its scalar type,
/// and compare.
template <typename V, typename Op>
void check_matches_scalar(Op op, char const *name) {
    using T         = simd::scalar_t<V>;
    constexpr int L = simd::lanes_v<V>;
    auto const    s = inputs<T>();
    for (size_t oa = 0; oa < s.size(); ++oa) {
        for (size_t ob = 0; ob < s.size(); ob += 2) {
            T a[L], b[L];
            for (int i = 0; i < L; ++i) {
                a[i] = s[(oa + static_cast<size_t>(i)) % s.size()];
                b[i] = s[(ob + 3 * static_cast<size_t>(i)) % s.size()];
            }
            auto const vector_result = op(simd::load<V>(a), simd::load<V>(b));
            using R                  = simd::scalar_t<std::remove_cvref_t<decltype(vector_result)>>;
            R out[L];
            simd::store(out, vector_result);
            for (int i = 0; i < L; ++i) {
                R const want = op(a[i], b[i]);
                INFO(name << " lane " << i << ": a = " << a[i] << ", b = " << b[i] << ", vector " << out[i] << ", scalar " << want);
                CHECK(same(out[i], want));
            }
        }
    }
}

} // namespace

// Each case runs for the native vectors, for an FP64 vector at FP32's width (the mixed-precision tier),
// and for a float vector four registers wide, so the wide forms are exercised for more than two parts.
// (Catch's type lists split on commas, so the wide types are named first.)
using DoubleAtFloatWidth = simd::Vec<double, simd::lanes<float>>;
using FloatFourWide      = simd::Vec<float, 4 * simd::lanes<float>>;
#define EINSUMS_GENERIC_VALUE_TYPES simd::Vec<float>, simd::Vec<double>, DoubleAtFloatWidth, FloatFourWide

TEMPLATE_TEST_CASE("arithmetic, mixed operators and compound assignment match the scalar instantiation", "[simd][generic]",
                   EINSUMS_GENERIC_VALUE_TYPES) {
    using V = TestType;
    using T = simd::scalar_t<V>;
    check_matches_scalar<V>([](auto x, auto y) { return x + y; }, "+");
    check_matches_scalar<V>([](auto x, auto y) { return x - y; }, "-");
    check_matches_scalar<V>([](auto x, auto y) { return x * y; }, "*");
    check_matches_scalar<V>([](auto x, auto y) { return x / y; }, "/");
    check_matches_scalar<V>([](auto x, auto y) { return -x + y; }, "unary -");
    check_matches_scalar<V>([](auto x, auto) { return x * 2; }, "v * 2");
    check_matches_scalar<V>([](auto x, auto) { return 3 - x; }, "3 - v");
    check_matches_scalar<V>([](auto x, auto) { return x / T(0.25); }, "v / T");
    check_matches_scalar<V>([](auto x, auto) { return T(0.5) + x; }, "T + v");
    check_matches_scalar<V>(
        [](auto x, auto y) {
            auto t = x;
            t += y;
            t *= 2;
            t -= T(0.75);
            t /= y;
            t *= x;
            t += 1;
            return t;
        },
        "compound");
}

TEMPLATE_TEST_CASE("math, fused forms and rounding match the scalar instantiation", "[simd][generic]", EINSUMS_GENERIC_VALUE_TYPES) {
    using V = TestType;
    using T = simd::scalar_t<V>;
    check_matches_scalar<V>([](auto x, auto y) { return simd::fmadd(x, y, x); }, "fmadd");
    check_matches_scalar<V>([](auto x, auto y) { return simd::fmsub(x, y, y); }, "fmsub");
    check_matches_scalar<V>([](auto x, auto y) { return simd::fnmadd(x, y, x); }, "fnmadd");
    check_matches_scalar<V>([](auto x, auto y) { return simd::fnmsub(y, x, y); }, "fnmsub");
    check_matches_scalar<V>([](auto x, auto y) { return simd::div(x, y); }, "div");
    check_matches_scalar<V>([](auto x, auto) { return simd::sqrt(x); }, "sqrt");
    check_matches_scalar<V>([](auto x, auto y) { return simd::min(x, y); }, "min");
    check_matches_scalar<V>([](auto x, auto y) { return simd::max(x, y); }, "max");
    check_matches_scalar<V>([](auto x, auto) { return simd::abs(x); }, "abs");
    check_matches_scalar<V>([](auto x, auto) { return simd::neg(x); }, "neg");
    check_matches_scalar<V>([](auto x, auto) { return simd::floor(x); }, "floor");
    check_matches_scalar<V>([](auto x, auto) { return simd::ceil(x); }, "ceil");
    check_matches_scalar<V>([](auto x, auto) { return simd::trunc(x); }, "trunc");
    check_matches_scalar<V>([](auto x, auto) { return simd::round(x); }, "round");
    check_matches_scalar<V>([](auto x, auto) { return simd::round_even(x); }, "round_even");
    check_matches_scalar<V>([](auto x, auto) { return simd::exp(x); }, "exp");
    check_matches_scalar<V>([](auto x, auto) { return simd::erf(x); }, "erf");
    check_matches_scalar<V>([](auto x, auto) { return simd::erfc(x); }, "erfc");
    check_matches_scalar<V>([](auto x, auto) { return simd::rsqrt(x); }, "rsqrt");
    check_matches_scalar<V>([](auto x, auto y) { return simd::exp(simd::fnmadd(x, y, x)); }, "exp of an expression");
}

TEMPLATE_TEST_CASE("compares, masks and select match the scalar instantiation", "[simd][generic]", EINSUMS_GENERIC_VALUE_TYPES) {
    using V = TestType;
    using T = simd::scalar_t<V>;
    check_matches_scalar<V>([](auto x, auto y) { return simd::select(simd::cmp_eq(x, y), x, y + 1); }, "cmp_eq");
    check_matches_scalar<V>([](auto x, auto y) { return simd::select(simd::cmp_ne(x, y), x, y + 1); }, "cmp_ne");
    check_matches_scalar<V>([](auto x, auto y) { return simd::select(simd::cmp_lt(x, y), x, y + 1); }, "cmp_lt");
    check_matches_scalar<V>([](auto x, auto y) { return simd::select(simd::cmp_le(x, y), x, y + 1); }, "cmp_le");
    check_matches_scalar<V>([](auto x, auto y) { return simd::select(simd::cmp_gt(x, y), x, y + 1); }, "cmp_gt");
    check_matches_scalar<V>([](auto x, auto y) { return simd::select(simd::cmp_ge(x, y), x, y + 1); }, "cmp_ge");
    check_matches_scalar<V>([](auto x, auto y) { return simd::select(!simd::cmp_lt(x, y), x, y + 1); }, "!m");
    check_matches_scalar<V>([](auto x, auto y) { return simd::select(!(simd::cmp_lt(x, y) | simd::cmp_ne(x, x)), y, x); }, "!(m | n)");
    check_matches_scalar<V>(
        [](auto x, auto y) {
            auto const lt = simd::cmp_lt(x, y), pos = simd::cmp_gt(x, simd::splat<decltype(x)>(T(0)));
            auto       r = simd::select(simd::bitwise_and(lt, pos), x, y);
            r            = simd::select(simd::bitwise_or(lt, pos), r, -r);
            r            = simd::select(simd::bitwise_xor(lt, pos), r * 2, r);
            return simd::select(simd::bitwise_andnot(lt, pos), r + 1, r);
        },
        "mask combinations");

    // any and all fold the lanes, so the vector answer is the fold of the scalar ones.
    constexpr int L = simd::lanes_v<V>;
    auto const    s = inputs<T>();
    for (size_t o = 0; o < s.size(); ++o) {
        T a[L], b[L];
        for (int i = 0; i < L; ++i) {
            a[i] = s[(o + static_cast<size_t>(i)) % s.size()];
            b[i] = s[(o + 7 * static_cast<size_t>(i)) % s.size()];
        }
        bool any_lane = false, every_lane = true;
        for (int i = 0; i < L; ++i) {
            any_lane   = any_lane || simd::any(simd::cmp_lt(a[i], b[i]));
            every_lane = every_lane && simd::all(simd::cmp_lt(a[i], b[i]));
        }
        CHECK(simd::any(simd::cmp_lt(simd::load<V>(a), simd::load<V>(b))) == any_lane);
        CHECK(simd::all(simd::cmp_lt(simd::load<V>(a), simd::load<V>(b))) == every_lane);
    }
}

TEST_CASE("convert matches the scalar instantiation", "[simd][generic]") {
    // In range everywhere. Out of range only on x86, where both give the type's minimum; NEON
    // saturates its vector conversion instead.
#if defined(__x86_64__) || defined(_M_X64)
    constexpr bool out_of_range_too = true;
#else
    constexpr bool out_of_range_too = false;
#endif
    auto const in_range = [](auto x, double limit) { return std::isfinite(x) && std::fabs(static_cast<double>(x)) < limit; };

    constexpr int F = simd::Vec<float>::lanes;
    for (float x : inputs<float>()) {
        if (!out_of_range_too && !in_range(x, 2147483520.0)) {
            continue;
        }
        int32_t out[F];
        simd::store(out, simd::convert<int32_t>(simd::splat<simd::Vec<float>>(x)));
        INFO("float " << x);
        CHECK(out[0] == simd::convert<int32_t>(x));
    }
    constexpr int D = simd::Vec<double>::lanes;
    for (double x : inputs<double>()) {
        if (!out_of_range_too && !in_range(x, 9.2e18)) {
            continue;
        }
        int64_t out[D];
        simd::store(out, simd::convert<int64_t>(simd::splat<simd::Vec<double>>(x)));
        INFO("double " << x);
        CHECK(out[0] == simd::convert<int64_t>(x));
    }
    for (int64_t x : {int64_t{0}, int64_t{-5}, int64_t{1} << 53, (int64_t{1} << 53) + 1, std::numeric_limits<int64_t>::max()}) {
        double out[D];
        simd::store(out, simd::convert<double>(simd::splat<simd::Vec<int64_t>>(x)));
        CHECK(same(out[0], simd::convert<double>(x)));
    }
    for (int32_t x : {0, -5, 16777217, std::numeric_limits<int32_t>::min()}) {
        float out[F];
        simd::store(out, simd::convert<float>(simd::splat<simd::Vec<int32_t>>(x)));
        CHECK(same(out[0], simd::convert<float>(x)));
    }
    CHECK(simd::convert<float>(1.0 + 0x1p-24) == 1.0f);
    CHECK(simd::convert<double>(1.5f) == 1.5);
}

namespace {

/// A tabulated function evaluated by a short Taylor series about the grid point below x, written once:
/// the shape of the Boys function interpolation.
template <typename V>
V interpolate(V x, simd::scalar_t<V> const *const (&taylor)[4], simd::scalar_t<V> inv_dx, simd::scalar_t<V> dx) {
    using S         = simd::scalar_t<V>;
    using I         = simd::gather_index_t<S>;
    V const    grid = simd::floor(x * inv_dx);
    auto const idx  = simd::convert<I>(grid);
    V const    d    = simd::fnmadd(grid, simd::splat<V>(dx), x); // x - grid * dx
    V          r    = simd::lookup(taylor[3], idx);
    r               = simd::fmadd(r, d, simd::lookup(taylor[2], idx));
    r               = simd::fmadd(r, d, simd::lookup(taylor[1], idx));
    return simd::fmadd(r, d, simd::lookup(taylor[0], idx));
}

} // namespace

TEMPLATE_TEST_CASE("a table interpolation written once gives the same lanes as its scalar instantiation", "[simd][generic]",
                   EINSUMS_GENERIC_VALUE_TYPES) {
    using V           = TestType;
    using T           = simd::scalar_t<V>;
    constexpr int  L  = simd::lanes_v<V>;
    T const        dx = T(0.125), inv_dx = T(8);
    std::vector<T> c0(400), c1(400), c2(400), c3(400);
    for (size_t k = 0; k < 400; ++k) {
        T const g = static_cast<T>(k) * dx;
        c0[k]     = std::exp(-g);
        c1[k]     = -c0[k];
        c2[k]     = c0[k] / 2;
        c3[k]     = -c0[k] / 6;
    }
    T const *const taylor[4] = {c0.data(), c1.data(), c2.data(), c3.data()};

    for (int start = 0; start < 40; ++start) {
        T x[L];
        for (int i = 0; i < L; ++i) {
            x[i] = T(0.013) * static_cast<T>((start * 97 + i * 31) % 3700); // below 48.1, inside the 400-point table
        }
        T out[L];
        simd::store(out, interpolate(simd::load<V>(x), taylor, inv_dx, dx));
        for (int i = 0; i < L; ++i) {
            T const want = interpolate(x[i], taylor, inv_dx, dx);
            INFO("x = " << x[i]);
            CHECK(same(out[i], want));
            CHECK_THAT(want, Catch::Matchers::WithinRel(std::exp(-x[i]), T(1e-4)));
        }
    }
}

namespace {
template <typename A, typename B>
concept multipliable = requires(A a, B b) { a * b; };
} // namespace

TEST_CASE("the generic layer's overloads resolve as designed", "[simd][generic]") {
    // Traits.
    STATIC_CHECK(simd::lanes_v<double> == 1);
    STATIC_CHECK(simd::lanes_v<simd::Vec<float>> == simd::Vec<float>::lanes);
    STATIC_CHECK(simd::is_vec_v<simd::Vec<double>>);
    STATIC_CHECK(!simd::is_vec_v<long double>);
    STATIC_CHECK(std::is_same_v<simd::scalar_t<simd::Vec<float>>, float>);
    STATIC_CHECK(std::is_same_v<simd::scalar_t<double>, double>);

    // A scalar mixes with a vector when it is the element type or an integer, never another
    // floating type, which would round differently in the scalar instantiation. The scalar fallback
    // build is the exception: there Vec<T> converts to and from T implicitly, so the built-in
    // operator applies.
    STATIC_CHECK(multipliable<simd::Vec<double>, int>);
    STATIC_CHECK(multipliable<long, simd::Vec<float>>);
    STATIC_CHECK(multipliable<simd::Vec<float>, float>);
    if constexpr (simd::native_bits > 0) {
        STATIC_CHECK(!multipliable<simd::Vec<float>, double>);
    }

    // Under using namespace, the C library still answers for scalars: the scalar overloads are
    // constrained templates, which never beat a plain function.
    {
        using namespace einsums::simd;
        CHECK(sqrt(2.0) == std::sqrt(2.0));
        CHECK(round(2.5) == 3.0);
        CHECK(abs(-3) == 3);
        CHECK(exp(1.0) == std::exp(1.0));
    }

    // The runtime ladder's select still resolves beside the scalar select.
    using Fn           = int (*)();
    static Fn const fn = simd::select<Fn>(+[] { return 7; }, nullptr, nullptr, nullptr, nullptr);
    CHECK(fn() == 7);
    CHECK(simd::select(true, 1.0, 2.0) == 1.0);
}

TEMPLATE_TEST_CASE("masked loadu, storeu and lookup match the scalar instantiation", "[simd][generic][mask]", EINSUMS_GENERIC_VALUE_TYPES) {
    // The vector run masks its lanes with a Mask<T, N>; the scalar run on lane i uses the bool for that lane.
    using V          = TestType;
    using T          = simd::scalar_t<V>;
    using I          = simd::gather_index_t<T>;
    constexpr int  L = simd::lanes_v<V>;
    std::vector<T> table(64);
    for (size_t k = 0; k < table.size(); ++k)
        table[k] = T(0.25) * static_cast<T>(k) - T(3);
    auto const s = inputs<T>();
    for (size_t o = 0; o < s.size(); ++o) {
        T x[L], y[L];
        I at[L];
        for (int i = 0; i < L; ++i) {
            x[i]  = s[(o + static_cast<size_t>(i)) % s.size()];
            y[i]  = s[(o + 3 * static_cast<size_t>(i) + 1) % s.size()];
            at[i] = static_cast<I>((o * 7 + static_cast<size_t>(i) * 13) % table.size());
        }
        auto const vm = simd::cmp_lt(simd::load<V>(x), simd::load<V>(y));
        T          loaded[L], looked[L], stored[L];
        simd::store(loaded, simd::loadu(table.data(), vm));
        simd::store(looked, simd::lookup(table.data(), simd::load<simd::Vec<I, L>>(at), vm));
        for (int i = 0; i < L; ++i)
            stored[i] = T(-1);
        simd::storeu(stored, simd::load<V>(x), vm);
        for (int i = 0; i < L; ++i) {
            bool const m = simd::cmp_lt(x[i], y[i]);
            INFO("lane " << i << ": x = " << x[i] << ", y = " << y[i]);
            CHECK(same(loaded[i], simd::loadu(table.data() + i, m)));
            CHECK(same(looked[i], simd::lookup(table.data(), at[i], m)));
            T scalar_stored = T(-1);
            simd::storeu(&scalar_stored, x[i], m);
            CHECK(same(stored[i], scalar_stored));
            CHECK(simd::count(m) == (m ? 1 : 0));
            CHECK(simd::none(m) == !m);
        }
    }
}
