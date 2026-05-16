//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <cstddef>

namespace einsums::simd {

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
    __m512i idx = _mm512_set_epi32(15 * stride, 14 * stride, 13 * stride, 12 * stride, 11 * stride, 10 * stride, 9 * stride, 8 * stride,
                                   7 * stride, 6 * stride, 5 * stride, 4 * stride, 3 * stride, 2 * stride, stride, 0);
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
    __m256i idx = _mm256_set_epi32(7 * stride, 6 * stride, 5 * stride, 4 * stride, 3 * stride, 2 * stride, stride, 0);
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
    __m512i idx = _mm512_set_epi32(15 * stride, 14 * stride, 13 * stride, 12 * stride, 11 * stride, 10 * stride, 9 * stride, 8 * stride,
                                   7 * stride, 6 * stride, 5 * stride, 4 * stride, 3 * stride, 2 * stride, stride, 0);
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

} // namespace einsums::simd
