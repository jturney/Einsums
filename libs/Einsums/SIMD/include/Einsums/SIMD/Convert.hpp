//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

EINSUMS_NAMESPACE_BEGIN(simd)
EINSUMS_SIMD_ISA_NAMESPACE_BEGIN()

// ===========================================================================
// Lane-wise conversion between element types of the same width, so the lane
// count carries over: convert<float>(Vec<int32_t>) and convert<int32_t>(Vec<float>).
//
// int32 to float rounds to nearest (ties to even), which is exact below 2^24.
// float to int32 truncates toward zero, as a C++ cast does. A NaN, or a value
// outside the int32 range, gives an unspecified lane: x86 returns INT32_MIN
// and NEON saturates. The scalar fallback returns INT32_MIN rather than
// performing the cast, which C++ leaves undefined.
// ===========================================================================

template <typename To, typename From>
EINSUMS_FORCEINLINE Vec<To> convert(Vec<From> v);
template <typename To, typename From>
EINSUMS_FORCEINLINE Vec<To> convert_low(Vec<From> v);
template <typename To, typename From>
EINSUMS_FORCEINLINE Vec<To> convert_high(Vec<From> v);
template <typename To, typename From>
EINSUMS_FORCEINLINE Vec<To> convert(Vec<From> low, Vec<From> high);

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, int32_t>(Vec<int32_t> v) {
    return _mm512_cvtepi32_ps(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, float>(Vec<float> v) {
    return _mm512_cvttps_epi32(v.reg);
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, int32_t>(Vec<int32_t> v) {
    return _mm256_cvtepi32_ps(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, float>(Vec<float> v) {
    return _mm256_cvttps_epi32(v.reg);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, int32_t>(Vec<int32_t> v) {
    return _mm_cvtepi32_ps(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, float>(Vec<float> v) {
    return _mm_cvttps_epi32(v.reg);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, int32_t>(Vec<int32_t> v) {
    return vcvtq_f32_s32(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, float>(Vec<float> v) {
    return vcvtq_s32_f32(v.reg);
}
#else
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, int32_t>(Vec<int32_t> v) {
    return {static_cast<float>(v.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, float>(Vec<float> v) {
    // 2^31 is exactly representable; every float below it and at or above -2^31 truncates into range.
    constexpr float limit = 2147483648.0f;
    if (!(v.reg >= -limit && v.reg < limit)) {
        return {std::numeric_limits<int32_t>::min()};
    }
    return {static_cast<int32_t>(v.reg)};
}
#endif

// ===========================================================================
// bitcast<To>(v): the same bits read as another element type of the same
// width, so the lane count carries over: Vec<float> and Vec<int32_t> or
// Vec<uint32_t>, Vec<double> and Vec<int64_t> or Vec<uint64_t>. It moves no
// data and compiles to nothing; it is how a kernel reaches a float's exponent
// and mantissa bits with the integer operations.
// ===========================================================================

template <typename To, typename From>
    requires(sizeof(To) == sizeof(From) && Vec<To>::lanes == Vec<From>::lanes)
EINSUMS_FORCEINLINE Vec<To> bitcast(Vec<From> v) {
    return std::bit_cast<typename Vec<To>::reg_type>(v.reg);
}

// ===========================================================================
// double and int64: convert<int64_t>(Vec<double>) and
// convert<double>(Vec<int64_t>), with the same lane count.
//
// double to int64 truncates toward zero, as a C++ cast does; a NaN, or a value
// outside the int64 range, gives INT64_MIN on x86 and saturates on NEON, as
// the int32 conversion above does. int64 to double rounds to nearest, ties to
// even, which is exact below 2^53.
//
// AVX-512DQ and NEON have instructions for both. Other x86 builds have none,
// so when every lane is below 2^51 in magnitude they use the bits of
// 1.5 * 2^52 + x, whose low mantissa bits hold x exactly, and otherwise they
// convert lane by lane. A table index, the common case, takes the fast path.
// ===========================================================================

namespace detail {
template <typename To, typename From>
EINSUMS_FORCEINLINE Vec<To> convert_lane_by_lane(Vec<From> v) {
    static_assert(Vec<To>::lanes == Vec<From>::lanes);
    alignas(native_alignment) From in[Vec<From>::lanes];
    alignas(native_alignment) To   out[Vec<To>::lanes];
    storea(in, v);
    for (int i = 0; i < Vec<From>::lanes; ++i) {
        if constexpr (std::is_integral_v<To>) {
            // 2^63 is exactly representable; every double below it and at or above -2^63 truncates into range.
            constexpr From limit = From{9223372036854775808.0};
            out[i]               = in[i] >= -limit && in[i] < limit ? static_cast<To>(in[i]) : std::numeric_limits<To>::min();
        } else {
            out[i] = static_cast<To>(in[i]);
        }
    }
    return loada(out);
}

#if (defined(__SSE2__) || defined(_M_X64)) && (!defined(__AVX__) || defined(__AVX2__)) && !defined(__AVX512DQ__)
/// 1.5 * 2^52: adding an integer below 2^51 in magnitude lands in [2^52, 2^53), where the ulp is one,
/// so the sum is exact and its bits are this constant's bits plus the integer.
inline constexpr double int64_magic = 6755399441055744.0;

EINSUMS_FORCEINLINE Vec<int64_t> double_to_int64(Vec<double> v) {
    Vec<double> const t = trunc(v);
    if (all(cmp_lt(abs(t), broadcast(2251799813685248.0)))) { // 2^51
        Vec<double> const magic = broadcast(int64_magic);
        return sub(bitcast<int64_t>(add(t, magic)), bitcast<int64_t>(magic));
    }
    return convert_lane_by_lane<int64_t>(v);
}

EINSUMS_FORCEINLINE Vec<double> int64_to_double(Vec<int64_t> v) {
    Vec<int64_t> const limit = broadcast(int64_t{2251799813685248}); // 2^51
    if (all(bitwise_and(cmp_lt(v, limit), cmp_gt(v, sub(broadcast(int64_t{0}), limit))))) {
        Vec<double> const magic = broadcast(int64_magic);
        return sub(bitcast<double>(add(v, bitcast<int64_t>(magic))), magic);
    }
    return convert_lane_by_lane<double>(v);
}
#endif
} // namespace detail

#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
template <>
EINSUMS_FORCEINLINE Vec<int64_t> convert<int64_t, double>(Vec<double> v) {
    return _mm512_cvttpd_epi64(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert<double, int64_t>(Vec<int64_t> v) {
    return _mm512_cvtepi64_pd(v.reg);
}
#elif defined(__AVX__) && !defined(__AVX2__)
// No 256-bit integer arithmetic for the fast path.
template <>
EINSUMS_FORCEINLINE Vec<int64_t> convert<int64_t, double>(Vec<double> v) {
    return detail::convert_lane_by_lane<int64_t>(v);
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert<double, int64_t>(Vec<int64_t> v) {
    return detail::convert_lane_by_lane<double>(v);
}
#elif defined(__SSE2__) || defined(_M_X64)
template <>
EINSUMS_FORCEINLINE Vec<int64_t> convert<int64_t, double>(Vec<double> v) {
    return detail::double_to_int64(v);
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert<double, int64_t>(Vec<int64_t> v) {
    return detail::int64_to_double(v);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<int64_t> convert<int64_t, double>(Vec<double> v) {
    return vcvtq_s64_f64(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert<double, int64_t>(Vec<int64_t> v) {
    return vcvtq_f64_s64(v.reg);
}
#else
template <>
EINSUMS_FORCEINLINE Vec<int64_t> convert<int64_t, double>(Vec<double> v) {
    return detail::convert_lane_by_lane<int64_t>(v);
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert<double, int64_t>(Vec<int64_t> v) {
    return detail::convert_lane_by_lane<double>(v);
}
#endif

// ===========================================================================
// float and double, and double and int32, where one vector holds twice the
// other's lanes. The shape is the 16-bit float conversions' below:
//
//   convert_low<double>(f)       lanes 0 .. L/2 - 1 of a Vec<float> or Vec<int32_t>, as double
//   convert_high<double>(f)      lanes L/2 .. L - 1, as double
//   convert<float>(low, high)    two Vec<double>, rounded to nearest even into one Vec<float>
//   convert<int32_t>(low, high)  two Vec<double>, truncated toward zero into one Vec<int32_t>
//
// Widening is exact. float rounding follows a C++ conversion: an overflow
// becomes an infinity and a NaN stays a NaN. The int32 truncation follows
// convert<int32_t>(Vec<float>) above, out-of-range lanes included.
//
// The scalar build has one lane of each type, so there it has same-lane
// conversions instead: convert<double>(Vec<float>) and so on.
// ===========================================================================

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <>
EINSUMS_FORCEINLINE Vec<double> convert_low<double, float>(Vec<float> v) {
    return _mm512_cvtps_pd(_mm512_castps512_ps256(v.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert_high<double, float>(Vec<float> v) {
    return _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v.reg), 1)));
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, double>(Vec<double> low, Vec<double> high) {
    __m256 const lo = _mm512_cvtpd_ps(low.reg);
    __m256 const hi = _mm512_cvtpd_ps(high.reg);
    return _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_castps_pd(lo)), _mm256_castps_pd(hi), 1));
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert_low<double, int32_t>(Vec<int32_t> v) {
    return _mm512_cvtepi32_pd(_mm512_castsi512_si256(v.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert_high<double, int32_t>(Vec<int32_t> v) {
    return _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(v.reg, 1));
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, double>(Vec<double> low, Vec<double> high) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvttpd_epi32(low.reg)), _mm512_cvttpd_epi32(high.reg), 1);
}
#elif defined(__AVX__)
// Every instruction here is AVX, so this needs no AVX2.
template <>
EINSUMS_FORCEINLINE Vec<double> convert_low<double, float>(Vec<float> v) {
    return _mm256_cvtps_pd(_mm256_castps256_ps128(v.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert_high<double, float>(Vec<float> v) {
    return _mm256_cvtps_pd(_mm256_extractf128_ps(v.reg, 1));
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, double>(Vec<double> low, Vec<double> high) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(low.reg)), _mm256_cvtpd_ps(high.reg), 1);
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert_low<double, int32_t>(Vec<int32_t> v) {
    return _mm256_cvtepi32_pd(_mm256_castsi256_si128(v.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert_high<double, int32_t>(Vec<int32_t> v) {
    return _mm256_cvtepi32_pd(_mm256_extractf128_si256(v.reg, 1));
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, double>(Vec<double> low, Vec<double> high) {
    return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm256_cvttpd_epi32(low.reg)), _mm256_cvttpd_epi32(high.reg), 1);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<double> convert_low<double, float>(Vec<float> v) {
    return _mm_cvtps_pd(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert_high<double, float>(Vec<float> v) {
    return _mm_cvtps_pd(_mm_movehl_ps(v.reg, v.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, double>(Vec<double> low, Vec<double> high) {
    return _mm_movelh_ps(_mm_cvtpd_ps(low.reg), _mm_cvtpd_ps(high.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert_low<double, int32_t>(Vec<int32_t> v) {
    return _mm_cvtepi32_pd(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert_high<double, int32_t>(Vec<int32_t> v) {
    return _mm_cvtepi32_pd(_mm_shuffle_epi32(v.reg, _MM_SHUFFLE(3, 2, 3, 2)));
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, double>(Vec<double> low, Vec<double> high) {
    return _mm_unpacklo_epi64(_mm_cvttpd_epi32(low.reg), _mm_cvttpd_epi32(high.reg));
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<double> convert_low<double, float>(Vec<float> v) {
    return vcvt_f64_f32(vget_low_f32(v.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert_high<double, float>(Vec<float> v) {
    return vcvt_high_f64_f32(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, double>(Vec<double> low, Vec<double> high) {
    return vcvt_high_f32_f64(vcvt_f32_f64(low.reg), high.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert_low<double, int32_t>(Vec<int32_t> v) {
    return vcvtq_f64_s64(vmovl_s32(vget_low_s32(v.reg)));
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert_high<double, int32_t>(Vec<int32_t> v) {
    return vcvtq_f64_s64(vmovl_high_s32(v.reg));
}
// FCVTZS saturates to int64; the saturating narrow carries that on to int32.
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, double>(Vec<double> low, Vec<double> high) {
    return vqmovn_high_s64(vqmovn_s64(vcvtq_s64_f64(low.reg)), vcvtq_s64_f64(high.reg));
}
#else
template <>
EINSUMS_FORCEINLINE Vec<double> convert<double, float>(Vec<float> v) {
    return {static_cast<double>(v.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, double>(Vec<double> v) {
    return {static_cast<float>(v.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<double> convert<double, int32_t>(Vec<int32_t> v) {
    return {static_cast<double>(v.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, double>(Vec<double> v) {
    constexpr double limit = 2147483648.0;
    if (!(v.reg > -limit - 1.0 && v.reg < limit)) {
        return {std::numeric_limits<int32_t>::min()};
    }
    return {static_cast<int32_t>(v.reg)};
}
#endif

// ===========================================================================
// Widening and narrowing between float and the 16-bit float types half_t and
// bfloat16_t, where the build has them (see Vec.hpp). A 16-bit vector holds
// exactly twice Vec<float>'s lanes, so it widens to two float vectors:
//
//   convert_low<float>(v)    lanes 0 .. L/2 - 1 of v, as float
//   convert_high<float>(v)   lanes L/2 .. L - 1 of v, as float
//   convert<H>(low, high)    low's lanes then high's, rounded to H
//
// Widening is exact. Narrowing rounds to nearest, ties to even, as a C++
// conversion does; a NaN stays a NaN, though its payload need not survive.
// NEON uses its conversion instructions. Every other build that has the types
// converts lane by lane through memory, which is correct everywhere and waits
// for hardware to measure a native version against.
// ===========================================================================

namespace detail {
/// Half of a 16-bit float vector, lanes [offset, offset + lanes of float), widened lane by lane.
template <typename H>
EINSUMS_FORCEINLINE Vec<float> widen_lanes(Vec<H> v, int offset) {
    constexpr int L = Vec<float>::lanes;
    static_assert(Vec<H>::lanes == 2 * L, "a 16-bit float vector holds twice float's lanes");
    alignas(native_alignment) H     narrow[2 * L];
    alignas(native_alignment) float wide[L];
    storeu(narrow, v);
    for (int i = 0; i < L; ++i) {
        wide[i] = static_cast<float>(narrow[offset + i]);
    }
    return loadu(wide);
}

/// Two float vectors rounded lane by lane into one 16-bit float vector.
template <typename H>
EINSUMS_FORCEINLINE Vec<H> narrow_lanes(Vec<float> low, Vec<float> high) {
    constexpr int L = Vec<float>::lanes;
    static_assert(Vec<H>::lanes == 2 * L, "a 16-bit float vector holds twice float's lanes");
    alignas(native_alignment) float wide[2 * L];
    alignas(native_alignment) H     narrow[2 * L];
    storeu(wide, low);
    storeu(wide + L, high);
    for (int i = 0; i < 2 * L; ++i) {
        narrow[i] = static_cast<H>(wide[i]);
    }
    return loadu(narrow);
}
} // namespace detail

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
#    if defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<float> convert_low<float, half_t>(Vec<half_t> v) {
    return vcvt_f32_f16(vget_low_f16(v.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert_high<float, half_t>(Vec<half_t> v) {
    return vcvt_high_f32_f16(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> convert<half_t, float>(Vec<float> low, Vec<float> high) {
    return vcvt_high_f16_f32(vcvt_f16_f32(low.reg), high.reg);
}
#    else
template <>
EINSUMS_FORCEINLINE Vec<float> convert_low<float, half_t>(Vec<half_t> v) {
    return detail::widen_lanes(v, 0);
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert_high<float, half_t>(Vec<half_t> v) {
    return detail::widen_lanes(v, Vec<float>::lanes);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> convert<half_t, float>(Vec<float> low, Vec<float> high) {
    return detail::narrow_lanes<half_t>(low, high);
}
#    endif
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
#    if defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<float> convert_low<float, bfloat16_t>(Vec<bfloat16_t> v) {
    return vcvtq_low_f32_bf16(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert_high<float, bfloat16_t>(Vec<bfloat16_t> v) {
    return vcvtq_high_f32_bf16(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<bfloat16_t> convert<bfloat16_t, float>(Vec<float> low, Vec<float> high) {
    return vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(low.reg), high.reg);
}
#    else
template <>
EINSUMS_FORCEINLINE Vec<float> convert_low<float, bfloat16_t>(Vec<bfloat16_t> v) {
    return detail::widen_lanes(v, 0);
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert_high<float, bfloat16_t>(Vec<bfloat16_t> v) {
    return detail::widen_lanes(v, Vec<float>::lanes);
}
template <>
EINSUMS_FORCEINLINE Vec<bfloat16_t> convert<bfloat16_t, float>(Vec<float> low, Vec<float> high) {
    return detail::narrow_lanes<bfloat16_t>(low, high);
}
#    endif
#endif

EINSUMS_SIMD_ISA_NAMESPACE_END()
EINSUMS_NAMESPACE_END(simd)
