//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/SIMD/Platform.hpp>

#include <cstring>
#include <type_traits>

// Include platform intrinsic headers
#if defined(__SSE2__) || defined(__AVX__) || defined(__AVX512F__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    if defined(_MSC_VER)
#        include <intrin.h>
#    else
#        include <immintrin.h>
#    endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#    include <arm_neon.h>
#endif

namespace einsums::simd {

// ---------------------------------------------------------------------------
// Half-precision type aliases.
//
// `half_t` is the IEEE-754 binary16 type (5-bit exponent, 10-bit mantissa)
// — same numeric range as float but ~half the precision. ARM uses `__fp16`
// (also exposed as `_Float16` on newer Clang) under the FP16 vector
// extension; x86 uses `_Float16` under AVX-512FP16.
//
// `bfloat16_t` is the brain-float format (8-bit exponent, 7-bit mantissa)
// — same range as float, much less precision. Useful for ML inference.
// Both ARM and x86 spell it `__bf16`.
//
// These aliases only exist when the compiler accepts the underlying type;
// builds without the appropriate -mcpu / -mavx512fp16 flags simply won't
// see them. Vec<half_t> / Vec<bfloat16_t> are correspondingly only defined
// on those builds.
// ---------------------------------------------------------------------------
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
using half_t = __fp16;
#elif defined(__AVX512FP16__)
using half_t = _Float16;
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
using bfloat16_t = __bf16;
#endif

// ---------------------------------------------------------------------------
// VecTraits<T> — maps scalar type T to native SIMD register type.
// Specializations are guarded by ISA macros. Widest available wins.
// ---------------------------------------------------------------------------

template <typename T>
struct VecTraits;

// ===========================================================================
// x86 AVX-512: 512-bit registers
// ===========================================================================
#if defined(__AVX512F__) && defined(__AVX512VL__)

template <>
struct VecTraits<float> {
    using reg_type             = __m512;
    using int_type             = __m512i;
    static constexpr int lanes = 16;
    static constexpr int bits  = 512;
};

template <>
struct VecTraits<double> {
    using reg_type             = __m512d;
    using int_type             = __m512i;
    static constexpr int lanes = 8;
    static constexpr int bits  = 512;
};

// All x86 integer types share __m512i — element type is encoded in the
// `lanes` count and the per-op intrinsic the user calls (epi32 / epi64 / …).
template <>
struct VecTraits<int32_t> {
    using reg_type             = __m512i;
    using int_type             = __m512i;
    static constexpr int lanes = 16;
    static constexpr int bits  = 512;
};
template <>
struct VecTraits<uint32_t> {
    using reg_type             = __m512i;
    using int_type             = __m512i;
    static constexpr int lanes = 16;
    static constexpr int bits  = 512;
};
template <>
struct VecTraits<int64_t> {
    using reg_type             = __m512i;
    using int_type             = __m512i;
    static constexpr int lanes = 8;
    static constexpr int bits  = 512;
};
template <>
struct VecTraits<uint64_t> {
    using reg_type             = __m512i;
    using int_type             = __m512i;
    static constexpr int lanes = 8;
    static constexpr int bits  = 512;
};

// 8-bit integer types — 64 lanes per 512-bit register. The general
// arithmetic API is intentionally NOT specialized for these because
// per-lane mul/shift on bytes has saturation surprises and many ISAs
// only expose dot-product accumulators rather than direct ops. Use
// the dot_product_* helpers in Operations.hpp.
template <>
struct VecTraits<int8_t> {
    using reg_type             = __m512i;
    using int_type             = __m512i;
    static constexpr int lanes = 64;
    static constexpr int bits  = 512;
};
template <>
struct VecTraits<uint8_t> {
    using reg_type             = __m512i;
    using int_type             = __m512i;
    static constexpr int lanes = 64;
    static constexpr int bits  = 512;
};

// AVX-512FP16 — 32 IEEE half-precision lanes per 512-bit register.
#    if defined(__AVX512FP16__)
template <>
struct VecTraits<half_t> {
    using reg_type             = __m512h;
    using int_type             = __m512i;
    static constexpr int lanes = 32;
    static constexpr int bits  = 512;
};
#    endif
// AVX-512BF16 — 32 bfloat16 lanes; native arithmetic intrinsics are limited
// to convert + dot product (see Operations.hpp). The register type alias
// follows Intel's `__m512bh`.
#    if defined(__AVX512BF16__)
template <>
struct VecTraits<bfloat16_t> {
    using reg_type             = __m512bh;
    using int_type             = __m512i;
    static constexpr int lanes = 32;
    static constexpr int bits  = 512;
};
#    endif

// ===========================================================================
// x86 AVX/AVX2: 256-bit registers
// ===========================================================================
#elif defined(__AVX__)

template <>
struct VecTraits<float> {
    using reg_type             = __m256;
    using int_type             = __m256i;
    static constexpr int lanes = 8;
    static constexpr int bits  = 256;
};

template <>
struct VecTraits<double> {
    using reg_type             = __m256d;
    using int_type             = __m256i;
    static constexpr int lanes = 4;
    static constexpr int bits  = 256;
};

template <>
struct VecTraits<int32_t> {
    using reg_type             = __m256i;
    using int_type             = __m256i;
    static constexpr int lanes = 8;
    static constexpr int bits  = 256;
};
template <>
struct VecTraits<uint32_t> {
    using reg_type             = __m256i;
    using int_type             = __m256i;
    static constexpr int lanes = 8;
    static constexpr int bits  = 256;
};
template <>
struct VecTraits<int64_t> {
    using reg_type             = __m256i;
    using int_type             = __m256i;
    static constexpr int lanes = 4;
    static constexpr int bits  = 256;
};
template <>
struct VecTraits<uint64_t> {
    using reg_type             = __m256i;
    using int_type             = __m256i;
    static constexpr int lanes = 4;
    static constexpr int bits  = 256;
};

template <>
struct VecTraits<int8_t> {
    using reg_type             = __m256i;
    using int_type             = __m256i;
    static constexpr int lanes = 32;
    static constexpr int bits  = 256;
};
template <>
struct VecTraits<uint8_t> {
    using reg_type             = __m256i;
    using int_type             = __m256i;
    static constexpr int lanes = 32;
    static constexpr int bits  = 256;
};

// ===========================================================================
// x86 SSE2: 128-bit registers
// ===========================================================================
#elif defined(__SSE2__) || (defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))

template <>
struct VecTraits<float> {
    using reg_type             = __m128;
    using int_type             = __m128i;
    static constexpr int lanes = 4;
    static constexpr int bits  = 128;
};

template <>
struct VecTraits<double> {
    using reg_type             = __m128d;
    using int_type             = __m128i;
    static constexpr int lanes = 2;
    static constexpr int bits  = 128;
};

template <>
struct VecTraits<int32_t> {
    using reg_type             = __m128i;
    using int_type             = __m128i;
    static constexpr int lanes = 4;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<uint32_t> {
    using reg_type             = __m128i;
    using int_type             = __m128i;
    static constexpr int lanes = 4;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<int64_t> {
    using reg_type             = __m128i;
    using int_type             = __m128i;
    static constexpr int lanes = 2;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<uint64_t> {
    using reg_type             = __m128i;
    using int_type             = __m128i;
    static constexpr int lanes = 2;
    static constexpr int bits  = 128;
};

template <>
struct VecTraits<int8_t> {
    using reg_type             = __m128i;
    using int_type             = __m128i;
    static constexpr int lanes = 16;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<uint8_t> {
    using reg_type             = __m128i;
    using int_type             = __m128i;
    static constexpr int lanes = 16;
    static constexpr int bits  = 128;
};

// ===========================================================================
// ARM NEON (aarch64, including Apple Silicon): 128-bit registers
// ===========================================================================
#elif defined(__aarch64__) || defined(_M_ARM64)

template <>
struct VecTraits<float> {
    using reg_type             = float32x4_t;
    using int_type             = int32x4_t;
    static constexpr int lanes = 4;
    static constexpr int bits  = 128;
};

template <>
struct VecTraits<double> {
    using reg_type             = float64x2_t;
    using int_type             = int64x2_t;
    static constexpr int lanes = 2;
    static constexpr int bits  = 128;
};

// NEON has distinct register types per signedness/width — vaddq_s32 vs
// vaddq_u32 vs vaddq_s64 etc — so each integer Vec specialization picks
// the matching `*x*_t` typedef.
template <>
struct VecTraits<int32_t> {
    using reg_type             = int32x4_t;
    using int_type             = int32x4_t;
    static constexpr int lanes = 4;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<uint32_t> {
    using reg_type             = uint32x4_t;
    using int_type             = uint32x4_t;
    static constexpr int lanes = 4;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<int64_t> {
    using reg_type             = int64x2_t;
    using int_type             = int64x2_t;
    static constexpr int lanes = 2;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<uint64_t> {
    using reg_type             = uint64x2_t;
    using int_type             = uint64x2_t;
    static constexpr int lanes = 2;
    static constexpr int bits  = 128;
};

template <>
struct VecTraits<int8_t> {
    using reg_type             = int8x16_t;
    using int_type             = int8x16_t;
    static constexpr int lanes = 16;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<uint8_t> {
    using reg_type             = uint8x16_t;
    using int_type             = uint8x16_t;
    static constexpr int lanes = 16;
    static constexpr int bits  = 128;
};

// NEON FP16 (M1+, gated on -march=armv8.2-a+fp16 or -mcpu=apple-m1+).
// 8 half-precision lanes per 128-bit register.
#    if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <>
struct VecTraits<half_t> {
    using reg_type             = float16x8_t;
    using int_type             = int16x8_t;
    static constexpr int lanes = 8;
    static constexpr int bits  = 128;
};
#    endif
// NEON BF16 (M3+ / Cortex-A715+, gated on -mcpu=apple-m3+ / -march=…+bf16).
#    if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
template <>
struct VecTraits<bfloat16_t> {
    using reg_type             = bfloat16x8_t;
    using int_type             = int16x8_t;
    static constexpr int lanes = 8;
    static constexpr int bits  = 128;
};
#    endif

// ===========================================================================
// Scalar fallback: no SIMD
// ===========================================================================
#else

template <>
struct VecTraits<float> {
    using reg_type             = float;
    using int_type             = int32_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 32;
};

template <>
struct VecTraits<double> {
    using reg_type             = double;
    using int_type             = int64_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 64;
};

template <>
struct VecTraits<int32_t> {
    using reg_type             = int32_t;
    using int_type             = int32_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 32;
};
template <>
struct VecTraits<uint32_t> {
    using reg_type             = uint32_t;
    using int_type             = uint32_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 32;
};
template <>
struct VecTraits<int64_t> {
    using reg_type             = int64_t;
    using int_type             = int64_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 64;
};
template <>
struct VecTraits<uint64_t> {
    using reg_type             = uint64_t;
    using int_type             = uint64_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 64;
};

template <>
struct VecTraits<int8_t> {
    using reg_type             = int8_t;
    using int_type             = int8_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 8;
};
template <>
struct VecTraits<uint8_t> {
    using reg_type             = uint8_t;
    using int_type             = uint8_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 8;
};

#endif

// ---------------------------------------------------------------------------
// Vec<T> — thin wrapper over a platform SIMD register.
//
// Design:
//   - Aggregate type, no virtual dispatch
//   - Implicitly converts to/from raw register for interop
//   - operator[] for debug/testing only (not performance-critical)
// ---------------------------------------------------------------------------

template <typename T>
struct Vec {
    // VecTraits<T> is only specialized for the types we support on the
    // active ISA tier. Instantiating Vec<T> for an unsupported type
    // (e.g. int8_t, char, std::complex) gives an "incomplete type"
    // error pointing here — the cleanest signal short of a `requires`
    // clause that doesn't tie us to a specific compile-time concept.
    using traits   = VecTraits<T>;
    using reg_type = typename traits::reg_type;

    static constexpr int lanes = traits::lanes;
    static constexpr int bits  = traits::bits;

    reg_type reg;

    Vec() = default;

    EINSUMS_FORCEINLINE Vec(reg_type r) : reg(r) {}

    EINSUMS_FORCEINLINE operator reg_type() const { return reg; }

    /// Element access (debug only — stores to temp buffer then indexes).
    EINSUMS_FORCEINLINE T operator[](int i) const {
        alignas(native_alignment) T buf[lanes];
        std::memcpy(buf, &reg, sizeof(reg));
        return buf[i];
    }
};

/// Convenience alias for the number of lanes in a Vec<T>.
template <typename T>
inline constexpr int lanes = Vec<T>::lanes;

} // namespace einsums::simd
