//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// rsqrt, erf and erfc (Math.hpp) against correctly rounded references: long double erfl, erfcl and
// 1 / sqrtl for double, and the double functions for float. The worst errors measured when they were
// written, over each function's whole range with subnormal results included:
//
//   rsqrt   1.49 ulp
//   erf     0.75 ulp below |x| = 0.5, 1.8 ulp above
//   erfc    1.1 ulp below x = 0.5, 4.0 ulp above (3.8 for float)
//
// and the bounds checked here are 1.6, 2.0 and 4.5 ulp. Special values are checked exactly. That the
// scalar instantiations match the vector lanes bit for bit is checked in GenericKernel.

#include <Einsums/SIMD/Math.hpp>

#include <cmath>
#include <limits>
#include <type_traits>

#include <catch2/catch_all.hpp>

namespace simd = einsums::simd;

namespace {

template <typename T>
double ulp_error(T got, long double ref) {
    if (std::isnan(got) || std::isinf(got)) {
        return 1e9;
    }
    T const r   = static_cast<T>(ref);
    T       ulp = std::nextafter(std::fabs(r), std::numeric_limits<T>::infinity()) - std::fabs(r);
    if (r == T(0)) {
        ulp = std::numeric_limits<T>::denorm_min();
    }
    return static_cast<double>(std::fabs(static_cast<long double>(got) - ref) / static_cast<long double>(ulp));
}

template <typename T>
long double ref_erf(T x) {
    if constexpr (std::is_same_v<T, double>) {
        return erfl(static_cast<long double>(x));
    } else {
        return static_cast<long double>(std::erf(static_cast<double>(x)));
    }
}
template <typename T>
long double ref_erfc(T x) {
    if constexpr (std::is_same_v<T, double>) {
        return erfcl(static_cast<long double>(x));
    } else {
        return static_cast<long double>(std::erfc(static_cast<double>(x)));
    }
}

/// Checks f against ref at n points evenly spaced in [lo, hi] (log-spaced when geometric is set),
/// run as vectors of V, to within bound ulp.
template <typename V, typename F, typename Ref>
void check_range(simd::scalar_t<V> lo, simd::scalar_t<V> hi, long n, F f, Ref ref, double bound, bool geometric = false) {
    using T         = simd::scalar_t<V>;
    constexpr int L = simd::lanes_v<V>;
    T             x[L], y[L];
    double        worst = 0;
    for (long i = 0; i < n; i += L) {
        for (int j = 0; j < L; ++j) {
            double const t = static_cast<double>(i + j) / static_cast<double>(n);
            x[j]           = geometric ? static_cast<T>(static_cast<double>(lo) * std::pow(static_cast<double>(hi / lo), t))
                                       : lo + (hi - lo) * static_cast<T>(t);
        }
        simd::store(y, f(simd::load<V>(x)));
        for (int j = 0; j < L; ++j) {
            double const e = ulp_error<T>(y[j], ref(x[j]));
            if (e > worst) {
                worst = e;
                INFO("x = " << x[j] << ", got " << y[j]);
                CHECK(e <= bound);
            }
        }
    }
}

template <typename T, typename F>
T lane0(F f, T x) {
    T out[simd::lanes<T>];
    simd::store(out, f(simd::splat<simd::Vec<T>>(x)));
    return out[0];
}

} // namespace

TEMPLATE_TEST_CASE("rsqrt is within 1.6 ulp of 1 / sqrt", "[simd][math][rsqrt]", float, double) {
    using T       = TestType;
    using lim     = std::numeric_limits<T>;
    auto const f  = [](auto v) { return simd::rsqrt(v); };
    T const    lo = std::is_same_v<T, double> ? T(1e-300) : T(1e-30);
    check_range<simd::Vec<T>>(lo, T(1) / lo, 1000000, f, [](T x) { return 1.0L / sqrtl(static_cast<long double>(x)); }, 1.6, true);
    CHECK(lane0<T>(f, T(4)) == T(0.5));
    CHECK(lane0<T>(f, T(0)) == lim::infinity());
    CHECK(lane0<T>(f, T(-0.0)) == -lim::infinity());
    CHECK(lane0<T>(f, lim::infinity()) == T(0));
    CHECK(std::isnan(lane0<T>(f, T(-1))));
    CHECK(std::isnan(lane0<T>(f, lim::quiet_NaN())));
}

TEMPLATE_TEST_CASE("erf is within 2 ulp of a correctly rounded erf", "[simd][math][erf]", float, double) {
    using T      = TestType;
    using lim    = std::numeric_limits<T>;
    using Wide   = simd::Vec<T, 2 * simd::lanes<float>>;
    auto const f = [](auto v) { return simd::erf(v); };
    auto const r = [](T x) { return ref_erf(x); };
    check_range<simd::Vec<T>>(T(-6), T(6), 1000000, f, r, 2.0);
    check_range<simd::Vec<T>>(T(-0.5), T(0.5), 400000, f, r, 2.0);
    check_range<simd::Vec<T>>(T(1e-30), T(0.5), 200000, f, r, 2.0, true);
    check_range<Wide>(T(-6), T(6), 200000, f, r, 2.0);
    CHECK(lane0<T>(f, T(0)) == T(0));
    CHECK(std::signbit(lane0<T>(f, T(-0.0))));
    CHECK(lane0<T>(f, lim::infinity()) == T(1));
    CHECK(lane0<T>(f, -lim::infinity()) == T(-1));
    CHECK(lane0<T>(f, T(100)) == T(1));
    CHECK(std::isnan(lane0<T>(f, lim::quiet_NaN())));
    // Odd, exactly.
    for (T x : {T(0.1), T(0.49), T(0.5), T(1.3), T(3.7)}) {
        CHECK(lane0<T>(f, -x) == -lane0<T>(f, x));
    }
}

TEMPLATE_TEST_CASE("erfc is within 4.5 ulp of a correctly rounded erfc", "[simd][math][erfc]", float, double) {
    using T        = TestType;
    using lim      = std::numeric_limits<T>;
    using Wide     = simd::Vec<T, 2 * simd::lanes<float>>;
    auto const f   = [](auto v) { return simd::erfc(v); };
    auto const r   = [](T x) { return ref_erfc(x); };
    T const    top = std::is_same_v<T, double> ? T(27.25) : T(10.05); // past here erfc is below the smallest subnormal
    check_range<simd::Vec<T>>(T(-6), T(0.5), 400000, f, r, 4.5);
    check_range<simd::Vec<T>>(T(0.5), top, 2000000, f, r, 4.5);
    check_range<Wide>(T(-6), top, 200000, f, r, 4.5);
    CHECK(lane0<T>(f, T(0)) == T(1));
    CHECK(lane0<T>(f, lim::infinity()) == T(0));
    CHECK(lane0<T>(f, -lim::infinity()) == T(2));
    CHECK(lane0<T>(f, T(100)) == T(0));
    CHECK(lane0<T>(f, T(-100)) == T(2));
    CHECK(std::isnan(lane0<T>(f, lim::quiet_NaN())));
    // Subnormal results are reached, not flushed.
    T const tiny = std::is_same_v<T, double> ? T(26.8) : T(9.6);
    CHECK(lane0<T>(f, tiny) > T(0));
    CHECK(lane0<T>(f, tiny) < lim::min());
}
