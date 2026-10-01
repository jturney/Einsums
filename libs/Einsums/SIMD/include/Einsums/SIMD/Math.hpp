//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/SIMD/Generic.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// ===========================================================================
// exp(x) for float and double: Vec<T>, wide Vec<T, N>, and plain T.
//
// One algorithm, written over the generic operations, so every instantiation
// computes the same result bit for bit (with the -ffp-contract caveat of
// Generic.hpp): a kernel's scalar reference matches its vector lanes here too.
//
//   1. Clamp x to a range just past where exp overflows and underflows, so
//      that the infinities and every value beyond reach those results through
//      the arithmetic below. The clamp keeps a NaN.
//   2. n = round_even(x log2(e)), and r = x - n ln(2), with ln(2) split in two
//      parts whose high part has enough trailing zero bits that n times it is
//      exact, with or without FMA; |r| <= ln(2) / 2.
//   3. e^r by its Taylor series: through r^13 / 13! for double and r^7 / 7!
//      for float, whose truncation error is far below an ulp; plain Horner
//      where FMA fuses, 1 + (r + r^2 q(r)) where it does not.
//   4. e^x = e^r 2^n, with 2^n built from exponent bits in two halves,
//      2^floor(n / 2) and 2^(n - floor(n / 2)), so each half is a normal
//      power of two and a result in the subnormal range rounds once, in the
//      last multiplication.
//
// The maximum error against a correctly rounded exp is measured in the unit
// tests (VecExp). It needs the 32- or 64-bit integer shift and add, which AVX
// without AVX2 does not have.
// ===========================================================================

EINSUMS_NAMESPACE_BEGIN(simd)
EINSUMS_SIMD_ISA_NAMESPACE_BEGIN()

namespace detail {
template <typename T>
struct exp_constants;

template <>
struct exp_constants<double> {
    using bits_type                       = int64_t;
    static constexpr int    mantissa_bits = 52;
    static constexpr double log2e         = 1.4426950408889634074;
    static constexpr double ln2_hi        = 6.93147180369123816490e-01; // 0x3FE62E42FEE00000: 21 trailing zero bits
    static constexpr double ln2_lo        = 1.90821492927058770002e-10;
    static constexpr double lowest        = -746.0;            // e^-746 is below half the smallest subnormal
    static constexpr double highest       = 710.0;             // e^710 overflows
    static constexpr double magic         = 0x1.8p52 + 1023.0; // 2^52 + 2^51 plus the exponent bias
    static constexpr int    terms         = 14;
    static constexpr double taylor[terms] = {
        1.0,          1.0,           1.0 / 2.0,      1.0 / 6.0,       1.0 / 24.0,       1.0 / 120.0,       1.0 / 720.0,
        1.0 / 5040.0, 1.0 / 40320.0, 1.0 / 362880.0, 1.0 / 3628800.0, 1.0 / 39916800.0, 1.0 / 479001600.0, 1.0 / 6227020800.0};
};

template <>
struct exp_constants<float> {
    using bits_type                      = int32_t;
    static constexpr int   mantissa_bits = 23;
    static constexpr float log2e         = 1.44269504088896341f;
    static constexpr float ln2_hi        = 0.693359375f; // 0x3F318000: 15 trailing zero bits
    static constexpr float ln2_lo        = -2.12194440e-4f;
    static constexpr float lowest        = -104.0f;
    static constexpr float highest       = 89.0f;
    static constexpr float magic         = 0x1.8p23f + 127.0f;
    static constexpr int   terms         = 8;
    static constexpr float taylor[terms] = {1.0f,         1.0f,          1.0f / 2.0f,   1.0f / 6.0f,
                                            1.0f / 24.0f, 1.0f / 120.0f, 1.0f / 720.0f, 1.0f / 5040.0f};
};

/// 2^k for integral k, from exponent bits: k + bias lands in the low mantissa bits of magic + k, and
/// the shift moves it into the exponent field and the rest of magic out of the word.
template <typename V>
EINSUMS_FORCEINLINE V power_of_two(V k) {
    using S           = scalar_t<V>;
    using C           = exp_constants<S>;
    auto const biased = bitcast<typename C::bits_type>(k + splat<V>(C::magic));
    return bitcast<S>(shift_left<C::mantissa_bits>(biased));
}

template <typename V>
concept floating_value = std::same_as<scalar_t<V>, float> || std::same_as<scalar_t<V>, double>;
} // namespace detail

/// e^x, lane by lane; see the top of this header.
template <typename V>
    requires(detail::floating_value<V> && (is_vec_v<V> || std::floating_point<V>))
EINSUMS_FORCEINLINE V exp(V x) {
    using S = scalar_t<V>;
    using C = detail::exp_constants<S>;

    // max(lowest, x) and min(highest, .) keep a NaN: each returns its second argument when the
    // comparison is false.
    V const xc = min(splat<V>(C::highest), max(splat<V>(C::lowest), x));

    V const n = round_even(xc * splat<V>(C::log2e));
    V       r = fnmadd(n, splat<V>(C::ln2_hi), xc);
    r         = fnmadd(n, splat<V>(C::ln2_lo), r);

    // Where the fused forms round once, plain Horner is the more accurate. Where they are a separate
    // multiply and add, summing as 1 + (r + r^2 q(r)), q the series from r^2 / 2! on divided by r^2,
    // keeps the rounding of the small terms out of the two largest. Measured against a correctly
    // rounded exp, either stays near one ulp at worst (VecExp).
    V p;
    if constexpr (scalar_fma_fused) {
        p = splat<V>(C::taylor[C::terms - 1]);
        for (int k = C::terms - 2; k >= 0; --k) {
            p = fmadd(p, r, splat<V>(C::taylor[k]));
        }
    } else {
        V q = splat<V>(C::taylor[C::terms - 1]);
        for (int k = C::terms - 2; k >= 2; --k) {
            q = fmadd(q, r, splat<V>(C::taylor[k]));
        }
        p = splat<V>(S(1)) + fmadd(r * r, q, r);
    }

    V const n1 = floor(n * S(0.5));
    V const n2 = n - n1;
    return p * detail::power_of_two(n1) * detail::power_of_two(n2);
}

// ===========================================================================
// rsqrt(x) = 1 / sqrt(x), as a division of the correctly rounded square root:
// within one ulp, and the same bits on every CPU. The hardware approximations
// (RSQRTPS, VRSQRT14PD and NEON's FRSQRTE) differ between vendors, so a result
// built on them would differ between machines and between a kernel's scalar
// and vector instantiations. rsqrt(+0) is +inf, rsqrt(-0) is -inf, a negative
// argument gives NaN and rsqrt(+inf) is +0.
// ===========================================================================

template <typename V>
    requires(detail::floating_value<V> && (is_vec_v<V> || std::floating_point<V>))
EINSUMS_FORCEINLINE V rsqrt(V x) {
    return div(splat<V>(scalar_t<V>(1)), sqrt(x));
}

// ===========================================================================
// erf(x) and erfc(x) for float and double: Vec<T>, wide Vec<T, N>, and plain
// T, one algorithm like exp, so every instantiation gives the same bits.
//
// By |x|:
//
//   |x| < 0.5      erf(x) = x + x E(x^2), E(s) = erf(x) / x - 1 a
//                  polynomial; erfc = (1 - x) - x E(x^2).
//   0.5 .. 2       erfc(|x|) = e^(-x^2) R(|x|), R(x) = erfc(x) e^(x^2) a
//   2 .. 4         polynomial in x - 1.25, and in x - 3, both exact.
//   4 .. top       erfc(|x|) = e^(-x^2) G(1 / x^2) / |x|, G a polynomial in
//                  1 / x^2 rescaled to [-1, 1]; past top erfc is below the
//                  smallest subnormal, and |x| is clamped there.
//
// erf = 1 - erfc above 0.5, and erf is odd and erfc(-x) = 2 - erfc(x). x^2 is
// split into an exact hi + lo (with FMA where it fuses, by Dekker's splitting
// where not) and e^(-x^2) taken as e^(-hi) (1 - lo): rounding x^2 alone would
// cost hundreds of ulps of e^(-x^2) at large x. A vector evaluates only the
// pieces its lanes fall in, selecting per lane.
//
// The polynomials are Chebyshev fits generated in 60-digit arithmetic, of
// degree 8, 19, 19 and 14 for double and 4, 10, 9 and 5 for float, by
// libs/Einsums/SIMD/devtool/generate_erf_coefficients.py, whose output is the
// two erf_constants specializations below. The error against a correctly
// rounded erf and erfc is measured in VecErf.
// ===========================================================================

namespace detail {
template <typename T>
struct erf_constants;

template <>
struct erf_constants<double> {
    // E(s) = erf(x) / x - 1 as a polynomial in s = x^2 over [0, 0.25], highest power first.
    static constexpr double small[9] = {0x1.8b4b60851826bp-20,  -0x1.f224dfc5409afp-17, 0x1.f98db34e8872ep-14,
                                        -0x1.c02d4f6e4921ap-11, 0x1.565bcbf8e0365p-8,   -0x1.b82ce30f2b28fp-6,
                                        0x1.ce2f21a03d814p-4,   -0x1.812746b0379b5p-2,  0x1.06eba8214db68p-3};
    // R(x) = erfc(x) e^(x^2) over [0.5, 2] in z = x - mid_center, exact, highest first.
    static constexpr double mid[20] = {-0x1.e6880b3e6e59fp-31, 0x1.d3beabe60ad52p-29, -0x1.63200e045b8c4p-27, 0x1.442fc009ca791p-25,
                                       -0x1.27857c651aa40p-23, 0x1.01c0e7cd302b4p-21, -0x1.b5b8f579866b9p-20, 0x1.6a15568a5632fp-18,
                                       -0x1.22fc4e1dac054p-16, 0x1.c57050abf75f8p-15, -0x1.55c07ce113526p-13, 0x1.f0fe6f32f5cd4p-12,
                                       -0x1.5b8bc94d17f85p-10, 0x1.d1b695ac27c52p-9,  -0x1.299636d6c488cp-7,  0x1.68a25a663ff1ep-6,
                                       -0x1.9b635ac624ae6p-5,  0x1.b56f45eef7e7bp-4,  -0x1.abaacdbfa8b07p-3,  0x1.78a692138767ap-2};
    // R(x) over [2, 4] in z = (x - high_center) / high_half.
    static constexpr double high[20] = {-0x1.765214aed4921p-42, 0x1.d497f9598b60ap-40, -0x1.cd539a8193ddep-38, 0x1.1666810f3d63bp-35,
                                        -0x1.527f395f218fap-33, 0x1.8c2f441a8c0efp-31, -0x1.c74e1c5371149p-29, 0x1.014e0a0208be0p-26,
                                        -0x1.1d7922347c7aap-24, 0x1.3699168d7bd7ep-22, -0x1.4b14624a945b9p-20, 0x1.595f1b10ffe11p-18,
                                        -0x1.6025103cace20p-16, 0x1.5e73930542870p-14, -0x1.53dec9d0889cap-12, 0x1.409cc2ed3f027p-10,
                                        -0x1.259061ba85698p-8,  0x1.043fe1a98c0d4p-6,  -0x1.bd6ae4d14b16fp-5,  0x1.6e9827d229d2dp-3};
    // x R(x) over [4, top] in w = (1 / x^2 - tail_center) / tail_half.
    static constexpr double tail[15]    = {0x1.6ae5899607639p-45, -0x1.8180f45f18194p-43, 0x1.5895d5aac5433p-41, -0x1.9eb09b32b75c9p-39,
                                           0x1.0cbf39bdce347p-36, -0x1.6dbf769c67944p-34, 0x1.0cc1890919272p-31, -0x1.b028361b59b95p-29,
                                           0x1.82286d574d16bp-26, -0x1.880166949423ap-23, 0x1.d280b2f57d42ep-20, -0x1.55715458a14a3p-16,
                                           0x1.4dc49a6929e40p-12, -0x1.028365bb3232bp-7,  0x1.1c75f6fd1ba00p-1};
    static constexpr double mid_center  = 0x1.4000000000000p+0;
    static constexpr double mid_half    = 0x1.0000000000000p+0;
    static constexpr double high_center = 0x1.8000000000000p+1;
    static constexpr double high_half   = 0x1.0000000000000p+0;
    static constexpr double tail_center = 0x1.056a8bf99179cp-5;
    static constexpr double tail_half   = 0x1.f52ae80cdd0c7p-6;
    static constexpr double top         = 0x1.b800000000000p+4;
    static constexpr double splitter    = 134217729.0; // 2^27 + 1, for an exact square without FMA
};

template <>
struct erf_constants<float> {
    // E(s) = erf(x) / x - 1 as a polynomial in s = x^2 over [0, 0.25], highest power first.
    static constexpr float small[5] = {0x1.3545120000000p-8f, -0x1.b666100000000p-6f, 0x1.ce25100000000p-4f, -0x1.8127320000000p-2f,
                                       0x1.06eba80000000p-3f};
    // R(x) = erfc(x) e^(x^2) over [0.5, 2] in z = x - mid_center, exact, highest first.
    static constexpr float mid[11] = {0x1.08ff7a0000000p-14f, -0x1.9398de0000000p-13f, 0x1.eb755c0000000p-12f, -0x1.5711380000000p-10f,
                                      0x1.d1e2be0000000p-9f,  -0x1.29b9f80000000p-7f,  0x1.68a1cc0000000p-6f,  -0x1.9b62740000000p-5f,
                                      0x1.b56f460000000p-4f,  -0x1.abaad00000000p-3f,  0x1.78a6920000000p-2f};
    // R(x) over [2, 4] in z = (x - high_center) / high_half.
    static constexpr float high[10] = {-0x1.7b892a0000000p-20f, 0x1.8e43dc0000000p-18f, -0x1.5d6cfa0000000p-16f, 0x1.5b7afe0000000p-14f,
                                       -0x1.53ee820000000p-12f, 0x1.40adf20000000p-10f, -0x1.2590420000000p-8f,  0x1.043fc00000000p-6f,
                                       -0x1.bd6ae40000000p-5f,  0x1.6e98280000000p-3f};
    // x R(x) over [4, top] in w = (1 / x^2 - tail_center) / tail_half.
    static constexpr float tail[6]     = {-0x1.60e6d00000000p-24f, 0x1.f07ca60000000p-21f, -0x1.a7eeee0000000p-17f,
                                          0x1.e812860000000p-13f,  -0x1.bafb840000000p-8f, 0x1.1bec2e0000000p-1f};
    static constexpr float mid_center  = 0x1.4000000000000p+0f;
    static constexpr float mid_half    = 0x1.0000000000000p+0f;
    static constexpr float high_center = 0x1.8000000000000p+1f;
    static constexpr float high_half   = 0x1.0000000000000p+0f;
    static constexpr float tail_center = 0x1.26fc800000000p-5f;
    static constexpr float tail_half   = 0x1.b207020000000p-6f;
    static constexpr float top         = 0x1.4800000000000p+3f;
    static constexpr float splitter    = 4097.0f; // 2^12 + 1, for an exact square without FMA
};

/// Horner over coefficients stored highest power first.
template <typename V, std::size_t K>
EINSUMS_FORCEINLINE V horner(V z, scalar_t<V> const (&c)[K]) {
    V p = splat<V>(c[0]);
    for (std::size_t k = 1; k < K; ++k) {
        p = fmadd(p, z, splat<V>(c[k]));
    }
    return p;
}

/// x^2 = hi + lo exactly: hi rounded, lo its error.
template <typename V>
EINSUMS_FORCEINLINE void exact_square(V x, V &hi, V &lo) {
    hi = x * x;
    if constexpr (scalar_fma_fused) {
        lo = fmsub(x, x, hi);
    } else {
        // Veltkamp's split into halves whose products are exact, then Dekker's sum of the errors.
        V const c  = x * splat<V>(erf_constants<scalar_t<V>>::splitter);
        V const xh = c - (c - x);
        V const xl = x - xh;
        lo         = ((xh * xh - hi) + (xh + xh) * xl) + xl * xl;
    }
}

/// erfc(a) for a >= 0.5, a no larger than the constants' top.
template <typename V>
EINSUMS_FORCEINLINE V erfc_tail(V a) {
    using S = scalar_t<V>;
    using C = erf_constants<S>;
    V hi, lo;
    exact_square(a, hi, lo);
    V const e = exp(-hi);
    V const g = fnmadd(e, lo, e); // e^(-hi) (1 - lo) = e^(-x^2) to within the series' first term

    auto const below_2 = cmp_lt(a, splat<V>(S(2)));
    auto const below_4 = cmp_lt(a, splat<V>(S(4)));
    V          r       = splat<V>(S(0));
    if (any(below_2)) {
        V const z = (a - splat<V>(C::mid_center)) * splat<V>(S(1) / C::mid_half);
        r         = select(below_2, horner(z, C::mid), r);
    }
    auto const in_high = bitwise_andnot(below_4, below_2); // not &: for bool masks that gives an int
    if (any(in_high)) {
        V const z = (a - splat<V>(C::high_center)) * splat<V>(S(1) / C::high_half);
        r         = select(in_high, horner(z, C::high), r);
    }
    if (any(!below_4)) {
        V const u = div(splat<V>(S(1)), hi);
        V const w = (u - splat<V>(C::tail_center)) * splat<V>(S(1) / C::tail_half);
        r         = select(!below_4, div(horner(w, C::tail), a), r);
    }
    return g * r;
}
} // namespace detail

/// erf(x), lane by lane; see above.
template <typename V>
    requires(detail::floating_value<V> && (is_vec_v<V> || std::floating_point<V>))
EINSUMS_FORCEINLINE V erf(V x) {
    using S          = scalar_t<V>;
    using C          = detail::erf_constants<S>;
    V const    a     = abs(x);
    auto const small = cmp_lt(a, splat<V>(S(0.5)));
    V          r     = splat<V>(S(0));
    if (any(small)) {
        // x + x E, with x added last and exactly once.
        r = select(small, fmadd(x, detail::horner(x * x, C::small), x), r);
    }
    if (any(!small)) {
        // min(top, a) keeps a NaN, which then reaches the result.
        V const t = splat<V>(S(1)) - detail::erfc_tail(min(splat<V>(C::top), a));
        r         = select(!small, select(cmp_lt(x, splat<V>(S(0))), -t, t), r);
    }
    return r;
}

/// erfc(x) = 1 - erf(x), lane by lane, accurate where erfc is small; see above.
template <typename V>
    requires(detail::floating_value<V> && (is_vec_v<V> || std::floating_point<V>))
EINSUMS_FORCEINLINE V erfc(V x) {
    using S          = scalar_t<V>;
    using C          = detail::erf_constants<S>;
    V const    a     = abs(x);
    auto const small = cmp_lt(a, splat<V>(S(0.5)));
    V          r     = splat<V>(S(0));
    if (any(small)) {
        // (1 - x) - x E: 1 - erf with its two largest terms taken first.
        r = select(small, fnmadd(x, detail::horner(x * x, C::small), splat<V>(S(1)) - x), r);
    }
    if (any(!small)) {
        V const t = detail::erfc_tail(min(splat<V>(C::top), a));
        r         = select(!small, select(cmp_lt(x, splat<V>(S(0))), splat<V>(S(2)) - t, t), r);
    }
    return r;
}

EINSUMS_SIMD_ISA_NAMESPACE_END()
EINSUMS_NAMESPACE_END(simd)
