//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// exp (Math.hpp) against a correctly rounded reference: expl in long double for double, and exp in
// double for float. Measured when it was written, the worst error over the whole range, subnormal
// results included, was 0.97 ulp for double and 1.03 ulp for float without FMA, 0.89 and 0.93 with
// it; the bound checked here is 1.1 ulp. Special values and the edges of the range are checked
// exactly, and the wide vector against the native one. That the scalar instantiation matches the
// vector lanes bit for bit is checked in GenericKernel.

#include <Einsums/SIMD/Math.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <catch2/catch_all.hpp>

namespace simd = einsums::simd;

namespace {

/// |got - ref| in units of the last place of ref rounded to T, with the subnormal spacing below the
/// normal range.
template <typename T>
double ulp_error(T got, long double ref) {
    if (ref > static_cast<long double>(std::numeric_limits<T>::max())) {
        return std::isinf(got) ? 0.0 : 1e9;
    }
    T const r   = static_cast<T>(ref);
    T       ulp = std::nextafter(std::fabs(r), std::numeric_limits<T>::infinity()) - std::fabs(r);
    if (r == T(0)) {
        ulp = std::numeric_limits<T>::denorm_min();
    }
    return static_cast<double>(std::fabs(static_cast<long double>(got) - ref) / static_cast<long double>(ulp));
}

template <typename T>
long double reference_exp(T x) {
    if constexpr (std::is_same_v<T, double>) {
        return expl(static_cast<long double>(x));
    } else {
        return static_cast<long double>(std::exp(static_cast<double>(x)));
    }
}

/// The worst error over n evenly spaced arguments in [lo, hi], run as vectors of V.
template <typename V>
double worst_error(simd::scalar_t<V> lo, simd::scalar_t<V> hi, long n) {
    using T             = simd::scalar_t<V>;
    constexpr int L     = simd::lanes_v<V>;
    double        worst = 0;
    T             x[L], y[L];
    for (long i = 0; i < n; i += L) {
        for (int j = 0; j < L; ++j) {
            x[j] = lo + (hi - lo) * static_cast<T>(static_cast<double>(i + j) / static_cast<double>(n));
        }
        simd::store(y, simd::exp(simd::load<V>(x)));
        for (int j = 0; j < L; ++j) {
            double const e = ulp_error<T>(y[j], reference_exp(x[j]));
            if (e > worst) {
                worst = e;
                INFO("x = " << x[j] << ", got " << y[j]);
                CHECK(e <= 1.1);
            }
        }
    }
    return worst;
}

template <typename T>
T exp_lane0(T x) {
    T out[simd::lanes<T>];
    simd::store(out, simd::exp(simd::splat<simd::Vec<T>>(x)));
    return out[0];
}

} // namespace

TEMPLATE_TEST_CASE("exp is within 1.1 ulp of a correctly rounded exp across its range", "[simd][math][exp]", float, double) {
    using T        = TestType;
    T const top    = std::is_same_v<T, double> ? T(709.78) : T(88.72);
    T const bottom = std::is_same_v<T, double> ? T(-745.1) : T(-103.9);
    T const sub    = std::is_same_v<T, double> ? T(-708.4) : T(-87.3); // below here the result is subnormal
    INFO("whole range: " << worst_error<simd::Vec<T>>(bottom, top, 2000000));
    INFO("[-1, 1]: " << worst_error<simd::Vec<T>>(T(-1), T(1), 400000));
    INFO("subnormal results: " << worst_error<simd::Vec<T>>(bottom, sub, 400000));
    using Wide = simd::Vec<T, 2 * simd::lanes<float>>;
    INFO("wide vector, whole range: " << worst_error<Wide>(bottom, top, 400000));
}

TEMPLATE_TEST_CASE("exp gives the exact results at the special values and the edges", "[simd][math][exp]", float, double) {
    using T       = TestType;
    using lim     = std::numeric_limits<T>;
    T const big   = std::is_same_v<T, double> ? T(1000) : T(200);
    T const over  = std::is_same_v<T, double> ? T(709.8) : T(88.8);    // just past overflow
    T const under = std::is_same_v<T, double> ? T(-745.2) : T(-104.0); // below half the smallest subnormal
    CHECK(exp_lane0(T(0)) == T(1));
    CHECK(exp_lane0(T(-0.0)) == T(1));
    CHECK(std::isnan(exp_lane0(lim::quiet_NaN())));
    CHECK(exp_lane0(lim::infinity()) == lim::infinity());
    CHECK(exp_lane0(-lim::infinity()) == T(0));
    CHECK(exp_lane0(big) == lim::infinity());
    CHECK(exp_lane0(over) == lim::infinity());
    CHECK(exp_lane0(-big) == T(0));
    CHECK(exp_lane0(under) == T(0));
    CHECK(ulp_error<T>(exp_lane0(T(1)), reference_exp(T(1))) <= 1.0);
    // The largest argument whose result is finite still is.
    T const finite_top = std::is_same_v<T, double> ? T(709.78) : T(88.72);
    CHECK(std::isfinite(exp_lane0(finite_top)));
    // The smallest subnormal result is reached, not flushed to zero.
    T const tiny = std::is_same_v<T, double> ? T(-744.4) : T(-103.2);
    CHECK(exp_lane0(tiny) > T(0));
    CHECK(exp_lane0(tiny) < lim::min());
}

TEMPLATE_TEST_CASE("exp of a wide vector is the native exp of each part", "[simd][math][exp]", float, double) {
    using T         = TestType;
    constexpr int N = 2 * simd::lanes<float>;
    T             x[N], wide[N], native[N];
    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<T>(i * 7 - 31) * T(1.37);
    }
    simd::store(wide, simd::exp(simd::load<simd::Vec<T, N>>(x)));
    for (int i = 0; i < N; i += simd::lanes<T>) {
        simd::store(native + i, simd::exp(simd::load<simd::Vec<T>>(x + i)));
    }
    for (int i = 0; i < N; ++i) {
        CHECK(wide[i] == native[i]);
    }
}
