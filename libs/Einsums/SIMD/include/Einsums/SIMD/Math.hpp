//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/SIMD/Generic.hpp>

#include <concepts>
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

EINSUMS_SIMD_ISA_NAMESPACE_END()
EINSUMS_NAMESPACE_END(simd)
