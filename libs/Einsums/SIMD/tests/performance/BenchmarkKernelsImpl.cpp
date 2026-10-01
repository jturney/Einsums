//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The benchmark kernels, compiled once per SIMD dispatch rung; see BenchmarkKernels.hpp.

#include <Einsums/SIMD/Generic.hpp>
#include <Einsums/SIMD/Math.hpp>
#include <Einsums/SIMD/Platform.hpp>

#include <cstddef>
#include <cstdint>

#include "BenchmarkKernels.hpp"

#if !defined(EINSUMS_SIMD_ARCH_NS)
#    define EINSUMS_SIMD_ARCH_NS arch_native
#endif

namespace simd_bench::EINSUMS_SIMD_ARCH_NS {

namespace simd = einsums::simd;

namespace {

template <typename V, typename I>
void gather_kernel(simd::scalar_t<V> const *table, I const *idx, simd::scalar_t<V> *out, std::size_t n) {
    constexpr int L = simd::lanes_v<V>;
    for (std::size_t i = 0; i < n; i += L) {
        simd::store(out + i, simd::lookup(table, simd::load<simd::Vec<I, L>>(idx + i)));
    }
}

template <typename To, typename From, int N>
void convert_kernel(From const *in, To *out, std::size_t n) {
    for (std::size_t i = 0; i < n; i += N) {
        simd::store(out + i, simd::convert<To>(simd::load<simd::Vec<From, N>>(in + i)));
    }
}

/// The interpolation body, written once for every value type V and index type I.
template <typename V, typename I>
V interpolate(Table<simd::scalar_t<V>> const &t, V x) {
    using S         = simd::scalar_t<V>;
    V const    grid = simd::floor(x * t.inv_dx);
    auto const g    = simd::convert<I>(grid);
    V const    d    = simd::fnmadd(grid, simd::splat<V>(t.dx), x);
    V          r    = simd::lookup(t.c[taylor_terms - 1], g);
    for (int k = taylor_terms - 2; k >= 0; --k) {
        r = simd::fmadd(r, d, simd::lookup(t.c[k], g));
    }
    return r;
}

template <typename V, typename I>
void interp_kernel(Table<simd::scalar_t<V>> const &t, simd::scalar_t<V> const *x, simd::scalar_t<V> *out, std::size_t n) {
    constexpr int L = simd::lanes_v<V>;
    for (std::size_t i = 0; i < n; i += L) {
        simd::store(out + i, interpolate<V, I>(t, simd::load<V>(x + i)));
    }
}

template <typename V>
void exp_kernel(simd::scalar_t<V> const *x, simd::scalar_t<V> *out, std::size_t n) {
    constexpr int L = simd::lanes_v<V>;
    for (std::size_t i = 0; i < n; i += L) {
        simd::store(out + i, simd::exp(simd::load<V>(x + i)));
    }
}

template <typename V, typename F>
void map_kernel(simd::scalar_t<V> const *x, simd::scalar_t<V> *out, std::size_t n, F f) {
    constexpr int L = simd::lanes_v<V>;
    for (std::size_t i = 0; i < n; i += L) {
        simd::store(out + i, f(simd::load<V>(x + i)));
    }
}

void erf_f64(double const *x, double *out, std::size_t n) {
    map_kernel<simd::Vec<double>>(x, out, n, [](auto v) { return simd::erf(v); });
}
void erfc_f64(double const *x, double *out, std::size_t n) {
    map_kernel<simd::Vec<double>>(x, out, n, [](auto v) { return simd::erfc(v); });
}
void erfc_f32(float const *x, float *out, std::size_t n) {
    map_kernel<simd::Vec<float>>(x, out, n, [](auto v) { return simd::erfc(v); });
}
void rsqrt_f64(double const *x, double *out, std::size_t n) {
    map_kernel<simd::Vec<double>>(x, out, n, [](auto v) { return simd::rsqrt(v); });
}

/// The hardware reciprocal square root estimate refined by Newton steps y (1.5 - 0.5 x y^2), for
/// comparison with rsqrt only: the estimate's bits differ between CPU vendors.
simd::Vec<double> rsqrt_estimate(simd::Vec<double> x) {
#if defined(__AVX512F__) && defined(__AVX512VL__)
    simd::Vec<double> y = _mm512_rsqrt14_pd(x.reg); // 14 bits: two steps reach double
    int constexpr steps = 2;
#elif defined(__AVX__)
    simd::Vec<double> y = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(x.reg))); // 12 bits
    int constexpr steps = 3;
#elif defined(__SSE2__)
    simd::Vec<double> y = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(x.reg)));
    int constexpr steps = 3;
#else
    simd::Vec<double> y = simd::rsqrt(x);
    int constexpr steps = 0;
#endif
    simd::Vec<double> const half_x = x * 0.5;
    for (int k = 0; k < steps; ++k) {
        y = y * simd::fnmadd(half_x * y, y, simd::splat<simd::Vec<double>>(1.5));
    }
    return y;
}
void rsqrt_f64_estimate(double const *x, double *out, std::size_t n) {
    map_kernel<simd::Vec<double>>(x, out, n, [](auto v) { return rsqrt_estimate(v); });
}

constexpr int LF = simd::lanes<float>;

} // namespace

Kernels const &kernels() noexcept {
    static constexpr Kernels table{
        .vector_bits         = einsums::simd::native_bits,
        .gather_f32          = &gather_kernel<simd::Vec<float>, int32_t>,
        .gather_f64          = &gather_kernel<simd::Vec<double>, int64_t>,
        .gather_f64_i32      = &gather_kernel<simd::Vec<double, LF>, int32_t>,
        .widen_f32_f64       = &convert_kernel<double, float, LF>,
        .narrow_f64_f32      = &convert_kernel<float, double, LF>,
        .f64_to_i64          = &convert_kernel<int64_t, double, simd::lanes<double>>,
        .f64_to_i32          = &convert_kernel<int32_t, double, LF>,
        .interp_f32          = &interp_kernel<simd::Vec<float>, int32_t>,
        .interp_f64          = &interp_kernel<simd::Vec<double>, int64_t>,
        .interp_f64_wide     = &interp_kernel<simd::Vec<double, LF>, int64_t>,
        .interp_f64_wide_i32 = &interp_kernel<simd::Vec<double, LF>, int32_t>,
        .interp_f64_scalar   = &interp_kernel<double, int64_t>,
        .exp_f32             = &exp_kernel<simd::Vec<float>>,
        .exp_f64             = &exp_kernel<simd::Vec<double>>,
        .exp_f64_wide        = &exp_kernel<simd::Vec<double, LF>>,
        .exp_f64_scalar      = &exp_kernel<double>,
        .erf_f64             = &erf_f64,
        .erfc_f64            = &erfc_f64,
        .erfc_f32            = &erfc_f32,
        .rsqrt_f64           = &rsqrt_f64,
        .rsqrt_f64_estimate  = &rsqrt_f64_estimate,
    };
    return table;
}

} // namespace simd_bench::EINSUMS_SIMD_ARCH_NS
