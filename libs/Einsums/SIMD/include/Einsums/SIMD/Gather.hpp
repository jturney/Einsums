//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

EINSUMS_NAMESPACE_BEGIN(simd)
EINSUMS_SIMD_ISA_NAMESPACE_BEGIN()

// ===========================================================================
// Gather: load Vec<T>::lanes elements from base[0], base[stride], base[2*stride], ...
//
// Uses hardware gather on AVX2, NEON structured loads for small strides,
// and a scalar loop as fallback.
// ===========================================================================

// ---------------------------------------------------------------------------
// Generic scalar fallback (used when no better option exists)
// ---------------------------------------------------------------------------
namespace detail {

template <typename T>
EINSUMS_FORCEINLINE Vec<T> gather_scalar(T const *base, std::ptrdiff_t stride) {
    alignas(native_alignment) T buf[Vec<T>::lanes];
    for (int i = 0; i < Vec<T>::lanes; ++i)
        buf[i] = base[i * stride];
    return loada(buf);
}

template <typename T>
EINSUMS_FORCEINLINE void scatter_scalar(T *base, std::ptrdiff_t stride, Vec<T> v) {
    alignas(native_alignment) T buf[Vec<T>::lanes];
    storea(buf, v);
    for (int i = 0; i < Vec<T>::lanes; ++i)
        base[i * stride] = buf[i];
}

/// Whether every offset 0, stride, ..., (lanes - 1) * stride of a Vec<T> fits the int32 lanes of a
/// 32-bit-index gather or scatter. A larger stride would wrap, so it takes the scalar loop instead.
template <typename T>
EINSUMS_FORCEINLINE bool offsets_fit_int32(std::ptrdiff_t stride) {
    constexpr std::ptrdiff_t limit = std::numeric_limits<int32_t>::max() / (Vec<T>::lanes - 1);
    return stride <= limit && stride >= -limit;
}

} // namespace detail

// ===========================================================================
// Gather dispatch
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> gather(T const *base, std::ptrdiff_t stride);

// ---------------------------------------------------------------------------
// x86 AVX-512: hardware gather with 512-bit registers
// ---------------------------------------------------------------------------
#if defined(__AVX512F__) && defined(__AVX512VL__)

template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    if (!detail::offsets_fit_int32<float>(stride))
        return detail::gather_scalar(base, stride);
    __m512i idx = _mm512_mullo_epi32(_mm512_set1_epi32(static_cast<int32_t>(stride)),
                                     _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
    return _mm512_i32gather_ps(idx, base, sizeof(float));
}

template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    __m512i idx = _mm512_set_epi64(7 * stride, 6 * stride, 5 * stride, 4 * stride, 3 * stride, 2 * stride, stride, 0);
    return _mm512_i64gather_pd(idx, base, sizeof(double));
}

#    if defined(__AVX512FP16__)
template <>
EINSUMS_FORCEINLINE Vec<half_t> gather(half_t const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    return detail::gather_scalar(base, stride);
}
#    endif

#    if defined(__AVX512BF16__)
template <>
EINSUMS_FORCEINLINE Vec<bfloat16_t> gather(bfloat16_t const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    return detail::gather_scalar(base, stride);
}
#    endif

// ---------------------------------------------------------------------------
// x86 AVX2: hardware gather with 256-bit registers
// ---------------------------------------------------------------------------
#elif defined(__AVX2__)

template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    if (!detail::offsets_fit_int32<float>(stride))
        return detail::gather_scalar(base, stride);
    __m256i idx = _mm256_mullo_epi32(_mm256_set1_epi32(static_cast<int32_t>(stride)), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
    return _mm256_i32gather_ps(base, idx, sizeof(float));
}

template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    // AVX2 _mm256_i64gather_pd takes __m256i for 4 int64 indices
    __m256i idx = _mm256_set_epi64x(3 * stride, 2 * stride, stride, 0);
    return _mm256_i64gather_pd(base, idx, sizeof(double));
}

// ---------------------------------------------------------------------------
// x86 AVX (no AVX2): no hardware gather, scalar fallback
// ---------------------------------------------------------------------------
#elif defined(__AVX__)

template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    return detail::gather_scalar(base, stride);
}

template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    return detail::gather_scalar(base, stride);
}

// ---------------------------------------------------------------------------
// x86 SSE2: scalar fallback
// ---------------------------------------------------------------------------
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)

template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    return detail::gather_scalar(base, stride);
}

template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    return detail::gather_scalar(base, stride);
}

// ---------------------------------------------------------------------------
// ARM NEON: structured loads for small strides, lane loads for general
// ---------------------------------------------------------------------------
#elif defined(__aarch64__) || defined(_M_ARM64)

template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    if (stride == 2)
        return vld2q_f32(base).val[0];
    if (stride == 3)
        return vld3q_f32(base).val[0];
    if (stride == 4)
        return vld4q_f32(base).val[0];
    // General fallback using lane loads
    float32x4_t r = vdupq_n_f32(0);
    r             = vld1q_lane_f32(base + 0 * stride, r, 0);
    r             = vld1q_lane_f32(base + 1 * stride, r, 1);
    r             = vld1q_lane_f32(base + 2 * stride, r, 2);
    r             = vld1q_lane_f32(base + 3 * stride, r, 3);
    return r;
}

template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    float64x2_t r = vdupq_n_f64(0);
    r             = vld1q_lane_f64(base + 0 * stride, r, 0);
    r             = vld1q_lane_f64(base + 1 * stride, r, 1);
    return r;
}

#    if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <>
EINSUMS_FORCEINLINE Vec<half_t> gather(half_t const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    if (stride == 2)
        return vld2q_f16(base).val[0];
    if (stride == 3)
        return vld3q_f16(base).val[0];
    if (stride == 4)
        return vld4q_f16(base).val[0];
    return detail::gather_scalar(base, stride);
}
#    endif

#    if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
template <>
EINSUMS_FORCEINLINE Vec<bfloat16_t> gather(bfloat16_t const *base, std::ptrdiff_t stride) {
    if (stride == 1)
        return loadu(base);
    if (stride == 2)
        return vld2q_bf16(base).val[0];
    if (stride == 3)
        return vld3q_bf16(base).val[0];
    if (stride == 4)
        return vld4q_bf16(base).val[0];
    return detail::gather_scalar(base, stride);
}
#    endif

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------
#else

template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, std::ptrdiff_t stride) {
    return {base[0]};
}

template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, std::ptrdiff_t stride) {
    return {base[0]};
}

#endif

// ===========================================================================
// Index gather: gather(base, idx) loads base[idx[0]], base[idx[1]], ... with
// one index per lane, counted in elements. The index vector has the element's
// width so the lane counts match: Vec<int32_t> for float, Vec<int64_t> for
// double. Every index must address a valid element; there is no masking.
//
// AVX2 and AVX-512 use their hardware gathers. Everything else, including
// NEON, which has no gather, reads lane by lane.
// ===========================================================================

// Empty for other types, so gather(base, stride) on them never trips over this overload.
template <typename T>
struct gather_index {};
template <>
struct gather_index<float> {
    using type = int32_t;
};
template <>
struct gather_index<double> {
    using type = int64_t;
};
/// The index element type gather(base, idx) takes for a T table.
template <typename T>
using gather_index_t = typename gather_index<T>::type;

template <typename T>
EINSUMS_FORCEINLINE Vec<T> gather(T const *base, Vec<gather_index_t<T>> idx);

namespace detail {
template <typename T>
EINSUMS_FORCEINLINE Vec<T> gather_index_scalar(T const *base, Vec<gather_index_t<T>> idx) {
    alignas(native_alignment) gather_index_t<T> at[Vec<T>::lanes];
    alignas(native_alignment) T                 buf[Vec<T>::lanes];
    storea(at, idx);
    for (int i = 0; i < Vec<T>::lanes; ++i)
        buf[i] = base[at[i]];
    return loada(buf);
}
} // namespace detail

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, Vec<int32_t> idx) {
    return _mm512_i32gather_ps(idx.reg, base, sizeof(float));
}
template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, Vec<int64_t> idx) {
    return _mm512_i64gather_pd(idx.reg, base, sizeof(double));
}
#elif defined(__AVX2__)
template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, Vec<int32_t> idx) {
    return _mm256_i32gather_ps(base, idx.reg, sizeof(float));
}
template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, Vec<int64_t> idx) {
    return _mm256_i64gather_pd(base, idx.reg, sizeof(double));
}
#elif !defined(__AVX__) && (defined(__x86_64__) || defined(_M_X64))
// SSE has no gather. Take the indices out of the register and build the result from scalar loads
// in registers: a round trip of the indices and the results through a stack buffer stalls on store
// forwarding, which made a float gather four times slower per element than a double one.
template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, Vec<int32_t> idx) {
    int const i0 = _mm_cvtsi128_si32(idx.reg);
    int const i1 = _mm_cvtsi128_si32(_mm_shuffle_epi32(idx.reg, _MM_SHUFFLE(1, 1, 1, 1)));
    int const i2 = _mm_cvtsi128_si32(_mm_shuffle_epi32(idx.reg, _MM_SHUFFLE(2, 2, 2, 2)));
    int const i3 = _mm_cvtsi128_si32(_mm_shuffle_epi32(idx.reg, _MM_SHUFFLE(3, 3, 3, 3)));
    return _mm_setr_ps(base[i0], base[i1], base[i2], base[i3]);
}
template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, Vec<int64_t> idx) {
    long long const i0 = _mm_cvtsi128_si64(idx.reg);
    long long const i1 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(idx.reg, idx.reg));
    return _mm_setr_pd(base[i0], base[i1]);
}
#else
template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, Vec<int32_t> idx) {
    return detail::gather_index_scalar(base, idx);
}
template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, Vec<int64_t> idx) {
    return detail::gather_index_scalar(base, idx);
}
#endif

// ===========================================================================
// Masked index gather: gather(base, idx, m) reads base[idx[i]] in each lane m
// sets and gives zero elsewhere. An inactive lane's index is never
// dereferenced, so it may be out of range. AVX2 and AVX-512 use their masked
// gathers; everything else reads the active lanes one by one.
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> gather(T const *base, Vec<gather_index_t<T>> idx, Mask<T> m);

namespace detail {
template <typename T>
EINSUMS_FORCEINLINE Vec<T> gather_index_masked_scalar(T const *base, Vec<gather_index_t<T>> idx, Mask<T> m) {
    alignas(native_alignment) gather_index_t<T> at[Vec<T>::lanes];
    alignas(native_alignment) T                 buf[Vec<T>::lanes] = {};
    storea(at, idx);
    uint64_t const set_lanes = to_bits(m);
    for (int i = 0; i < Vec<T>::lanes; ++i) {
        if ((set_lanes >> i) & 1u) {
            buf[i] = base[at[i]];
        }
    }
    return loada(buf);
}
} // namespace detail

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, Vec<int32_t> idx, Mask<float> m) {
    return _mm512_mask_i32gather_ps(_mm512_setzero_ps(), m.reg, idx.reg, base, sizeof(float));
}
template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, Vec<int64_t> idx, Mask<double> m) {
    return _mm512_mask_i64gather_pd(_mm512_setzero_pd(), m.reg, idx.reg, base, sizeof(double));
}
#elif defined(__AVX2__)
template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, Vec<int32_t> idx, Mask<float> m) {
    return _mm256_mask_i32gather_ps(_mm256_setzero_ps(), base, idx.reg, m.reg, sizeof(float));
}
template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, Vec<int64_t> idx, Mask<double> m) {
    return _mm256_mask_i64gather_pd(_mm256_setzero_pd(), base, idx.reg, m.reg, sizeof(double));
}
#else
template <>
EINSUMS_FORCEINLINE Vec<float> gather(float const *base, Vec<int32_t> idx, Mask<float> m) {
    return detail::gather_index_masked_scalar(base, idx, m);
}
template <>
EINSUMS_FORCEINLINE Vec<double> gather(double const *base, Vec<int64_t> idx, Mask<double> m) {
    return detail::gather_index_masked_scalar(base, idx, m);
}
#endif

// ===========================================================================
// Scatter: store Vec<T>::lanes elements to base[0], base[stride], ...
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE void scatter(T *base, std::ptrdiff_t stride, Vec<T> v);

// ---------------------------------------------------------------------------
// x86 AVX-512: hardware scatter
// ---------------------------------------------------------------------------
#if defined(__AVX512F__) && defined(__AVX512VL__)

template <>
EINSUMS_FORCEINLINE void scatter(float *base, std::ptrdiff_t stride, Vec<float> v) {
    if (stride == 1) {
        storeu(base, v);
        return;
    }
    if (!detail::offsets_fit_int32<float>(stride)) {
        detail::scatter_scalar(base, stride, v);
        return;
    }
    __m512i idx = _mm512_mullo_epi32(_mm512_set1_epi32(static_cast<int32_t>(stride)),
                                     _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
    _mm512_i32scatter_ps(base, idx, v.reg, sizeof(float));
}

template <>
EINSUMS_FORCEINLINE void scatter(double *base, std::ptrdiff_t stride, Vec<double> v) {
    if (stride == 1) {
        storeu(base, v);
        return;
    }
    __m512i idx = _mm512_set_epi64(7 * stride, 6 * stride, 5 * stride, 4 * stride, 3 * stride, 2 * stride, stride, 0);
    _mm512_i64scatter_pd(base, idx, v.reg, sizeof(double));
}

#    if defined(__AVX512FP16__)
template <>
EINSUMS_FORCEINLINE void scatter(half_t *base, std::ptrdiff_t stride, Vec<half_t> v) {
    if (stride == 1) {
        storeu(base, v);
        return;
    }
    detail::scatter_scalar(base, stride, v);
}
#    endif

#    if defined(__AVX512BF16__)
template <>
EINSUMS_FORCEINLINE void scatter(bfloat16_t *base, std::ptrdiff_t stride, Vec<bfloat16_t> v) {
    if (stride == 1) {
        storeu(base, v);
        return;
    }
    detail::scatter_scalar(base, stride, v);
}
#    endif

// ---------------------------------------------------------------------------
// All other platforms: scalar scatter fallback
// ---------------------------------------------------------------------------
#else

template <>
EINSUMS_FORCEINLINE void scatter(float *base, std::ptrdiff_t stride, Vec<float> v) {
    if (stride == 1) {
        storeu(base, v);
        return;
    }
    detail::scatter_scalar(base, stride, v);
}

template <>
EINSUMS_FORCEINLINE void scatter(double *base, std::ptrdiff_t stride, Vec<double> v) {
    if (stride == 1) {
        storeu(base, v);
        return;
    }
    detail::scatter_scalar(base, stride, v);
}

#    if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <>
EINSUMS_FORCEINLINE void scatter(half_t *base, std::ptrdiff_t stride, Vec<half_t> v) {
    if (stride == 1) {
        storeu(base, v);
        return;
    }
    detail::scatter_scalar(base, stride, v);
}
#    endif

#    if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
template <>
EINSUMS_FORCEINLINE void scatter(bfloat16_t *base, std::ptrdiff_t stride, Vec<bfloat16_t> v) {
    if (stride == 1) {
        storeu(base, v);
        return;
    }
    detail::scatter_scalar(base, stride, v);
}
#    endif

#endif

// ===========================================================================
// Compile-time stride overloads: gather_fixed<Stride> / scatter_fixed<Stride>
//
// When the stride is a compile-time constant, the compiler can:
//   - Eliminate branch chains (no runtime if/else on stride value)
//   - Select the optimal NEON structured load (vld2q/vld3q/vld4q) directly
//   - Fold the stride into addressing math
//
// Usage: gather_fixed<1>(ptr) instead of gather(ptr, 1)
// ===========================================================================

template <std::ptrdiff_t Stride, typename T>
EINSUMS_FORCEINLINE Vec<T> gather_fixed(T const *base) {
    if constexpr (Stride == 1) {
        return loadu(base);
    } else {
#if defined(__aarch64__) || defined(_M_ARM64)
        // NEON structured loads for known small strides
        if constexpr (std::is_same_v<T, float>) {
            if constexpr (Stride == 2)
                return vld2q_f32(base).val[0];
            else if constexpr (Stride == 3)
                return vld3q_f32(base).val[0];
            else if constexpr (Stride == 4)
                return vld4q_f32(base).val[0];
            else
                return detail::gather_scalar(base, Stride);
        } else {
            return detail::gather_scalar(base, Stride);
        }
#elif defined(__AVX2__)
        return gather(base, Stride); // AVX2 hardware gather handles any stride
#else
        return detail::gather_scalar(base, Stride);
#endif
    }
}

template <std::ptrdiff_t Stride, typename T>
EINSUMS_FORCEINLINE void scatter_fixed(T *base, Vec<T> v) {
    if constexpr (Stride == 1) {
        storeu(base, v);
    } else {
#if defined(__AVX512F__) && defined(__AVX512VL__)
        scatter(base, Stride, v); // AVX-512 hardware scatter
#else
        detail::scatter_scalar(base, Stride, v);
#endif
    }
}

EINSUMS_SIMD_ISA_NAMESPACE_END()
EINSUMS_NAMESPACE_END(simd)
