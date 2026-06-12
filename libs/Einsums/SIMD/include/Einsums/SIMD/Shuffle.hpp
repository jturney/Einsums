//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/SIMD/ComplexVec.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>

namespace einsums::simd {

// ===========================================================================
// In-register transpose: transpose_inplace(Vec<T> *rows)
//
// Transposes an N×N block where N = Vec<T>::lanes. The caller passes a
// pointer to N Vec<T> values representing the rows of the block. On return,
// rows[i] contains what was column i of the original block.
//
// The block size adapts to the native register width automatically:
//   AVX-512 float: 16×16    AVX-512 double: 8×8
//   AVX float:     8×8      AVX double:     4×4
//   SSE2 float:    4×4      SSE2 double:    2×2
//   NEON float:    4×4      NEON double:    2×2
//   Scalar:        1×1 (no-op)
// ===========================================================================

// ---------------------------------------------------------------------------
// x86 AVX-512: 512-bit registers
// ---------------------------------------------------------------------------
#if defined(__AVX512F__) && defined(__AVX512VL__)

// 8×8 double transpose (AVX-512)
EINSUMS_FORCEINLINE void transpose_inplace(Vec<double> *rows) {
    // Phase 1: 2×2 transposes within 128-bit lanes
    for (int i = 0; i < 8; i += 2) {
        auto lo     = _mm512_unpacklo_pd(rows[i], rows[i + 1]);
        auto hi     = _mm512_unpackhi_pd(rows[i], rows[i + 1]);
        rows[i]     = lo;
        rows[i + 1] = hi;
    }
    // Phase 2: 4×4 permute across 256-bit lanes
    for (int i = 0; i < 4; i++) {
        auto a      = _mm512_shuffle_f64x2(rows[i], rows[i + 4], 0x88);
        auto b      = _mm512_shuffle_f64x2(rows[i], rows[i + 4], 0xdd);
        rows[i]     = a;
        rows[i + 4] = b;
    }
    // Phase 3: final 128-bit lane swap
    auto t0 = rows[0];
    auto t1 = rows[1];
    auto t2 = rows[2];
    auto t3 = rows[3];
    auto t4 = rows[4];
    auto t5 = rows[5];
    auto t6 = rows[6];
    auto t7 = rows[7];
    rows[0] = _mm512_shuffle_f64x2(t0, t2, 0x88);
    rows[1] = _mm512_shuffle_f64x2(t1, t3, 0x88);
    rows[2] = _mm512_shuffle_f64x2(t0, t2, 0xdd);
    rows[3] = _mm512_shuffle_f64x2(t1, t3, 0xdd);
    rows[4] = _mm512_shuffle_f64x2(t4, t6, 0x88);
    rows[5] = _mm512_shuffle_f64x2(t5, t7, 0x88);
    rows[6] = _mm512_shuffle_f64x2(t4, t6, 0xdd);
    rows[7] = _mm512_shuffle_f64x2(t5, t7, 0xdd);
}

// 16×16 float transpose (AVX-512)
//
// Four-phase algorithm operating on 16 __m512 registers:
//   Phase 1: unpacklo/hi interleaves pairs within 128-bit lanes
//   Phase 2: shuffle_ps groups 4 elements within each 128-bit lane
//   Phase 3: shuffle_f32x4 permutes 128-bit lanes, combining 4-row blocks into 8-row blocks
//   Phase 4: shuffle_f32x4 permutes 128-bit lanes, combining 8-row blocks into the 16-row result
//
// After phases 1+2, each register s[k] contains, in its four 128-bit lanes,
// 4 elements of column (k, k+4, k+8, k+12) from a group of 4 consecutive rows.
// Phases 3+4 reassemble these lane fragments into complete 16-element columns.
EINSUMS_FORCEINLINE void transpose_inplace(Vec<float> *rows) {
    // Phase 1+2: Process 4 groups of 4 rows each.
    // After this, rows[g*4+k] holds column-group k from row-group g.
    for (int g = 0; g < 4; g++) {
        auto *r  = &rows[g * 4]; // NOLINT
        auto  t0 = _mm512_unpacklo_ps(r[0], r[1]);
        auto  t1 = _mm512_unpackhi_ps(r[0], r[1]);
        auto  t2 = _mm512_unpacklo_ps(r[2], r[3]);
        auto  t3 = _mm512_unpackhi_ps(r[2], r[3]);
        r[0]     = _mm512_shuffle_ps(t0, t2, 0x44);
        r[1]     = _mm512_shuffle_ps(t0, t2, 0xee);
        r[2]     = _mm512_shuffle_ps(t1, t3, 0x44);
        r[3]     = _mm512_shuffle_ps(t1, t3, 0xee);
    }

    // Phase 3+4: For each column-group k (0..3), combine the 4 row-groups
    // into 4 complete output columns (k, k+4, k+8, k+12).
    for (int k = 0; k < 4; k++) {
        // Phase 3: merge row-groups A(0-3)+B(4-7) and C(8-11)+D(12-15)
        auto uAB_lo = _mm512_shuffle_f32x4(rows[k], rows[4 + k], 0x88);
        auto uAB_hi = _mm512_shuffle_f32x4(rows[k], rows[4 + k], 0xdd);
        auto uCD_lo = _mm512_shuffle_f32x4(rows[8 + k], rows[12 + k], 0x88);
        auto uCD_hi = _mm512_shuffle_f32x4(rows[8 + k], rows[12 + k], 0xdd);

        // Phase 4: merge AB+CD into complete columns
        rows[k]      = _mm512_shuffle_f32x4(uAB_lo, uCD_lo, 0x88); // column k
        rows[k + 4]  = _mm512_shuffle_f32x4(uAB_hi, uCD_hi, 0x88); // column k+4
        rows[k + 8]  = _mm512_shuffle_f32x4(uAB_lo, uCD_lo, 0xdd); // column k+8
        rows[k + 12] = _mm512_shuffle_f32x4(uAB_hi, uCD_hi, 0xdd); // column k+12
    }
}

// ---------------------------------------------------------------------------
// x86 AVX/AVX2: 256-bit registers
// ---------------------------------------------------------------------------
#elif defined(__AVX__)

// 4×4 double transpose (AVX)
EINSUMS_FORCEINLINE void transpose_inplace(Vec<double> *rows) {
    auto t0 = _mm256_shuffle_pd(rows[0], rows[1], 0x0); // a0 b0 a2 b2
    auto t1 = _mm256_shuffle_pd(rows[0], rows[1], 0xf); // a1 b1 a3 b3
    auto t2 = _mm256_shuffle_pd(rows[2], rows[3], 0x0); // c0 d0 c2 d2
    auto t3 = _mm256_shuffle_pd(rows[2], rows[3], 0xf); // c1 d1 c3 d3
    rows[0] = _mm256_permute2f128_pd(t0, t2, 0x20);
    rows[1] = _mm256_permute2f128_pd(t1, t3, 0x20);
    rows[2] = _mm256_permute2f128_pd(t0, t2, 0x31);
    rows[3] = _mm256_permute2f128_pd(t1, t3, 0x31);
}

// 8×8 float transpose (AVX)
EINSUMS_FORCEINLINE void transpose_inplace(Vec<float> *rows) {
    // Phase 1: interleave pairs
    auto t0 = _mm256_unpacklo_ps(rows[0], rows[1]);
    auto t1 = _mm256_unpackhi_ps(rows[0], rows[1]);
    auto t2 = _mm256_unpacklo_ps(rows[2], rows[3]);
    auto t3 = _mm256_unpackhi_ps(rows[2], rows[3]);
    auto t4 = _mm256_unpacklo_ps(rows[4], rows[5]);
    auto t5 = _mm256_unpackhi_ps(rows[4], rows[5]);
    auto t6 = _mm256_unpacklo_ps(rows[6], rows[7]);
    auto t7 = _mm256_unpackhi_ps(rows[6], rows[7]);

    // Phase 2: 2×2 block shuffle
    auto s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    auto s1 = _mm256_shuffle_ps(t0, t2, 0xee);
    auto s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    auto s3 = _mm256_shuffle_ps(t1, t3, 0xee);
    auto s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    auto s5 = _mm256_shuffle_ps(t4, t6, 0xee);
    auto s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    auto s7 = _mm256_shuffle_ps(t5, t7, 0xee);

    // Phase 3: cross-lane permute
    rows[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    rows[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    rows[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    rows[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    rows[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    rows[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    rows[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    rows[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// ---------------------------------------------------------------------------
// x86 SSE2: 128-bit registers
// ---------------------------------------------------------------------------
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)

// 2×2 double transpose (SSE2)
EINSUMS_FORCEINLINE void transpose_inplace(Vec<double> *rows) {
    auto lo = _mm_unpacklo_pd(rows[0], rows[1]);
    auto hi = _mm_unpackhi_pd(rows[0], rows[1]);
    rows[0] = lo;
    rows[1] = hi;
}

// 4×4 float transpose (SSE2)
EINSUMS_FORCEINLINE void transpose_inplace(Vec<float> *rows) {
    auto t0 = _mm_unpacklo_ps(rows[0], rows[1]); // a0 b0 a1 b1
    auto t1 = _mm_unpackhi_ps(rows[0], rows[1]); // a2 b2 a3 b3
    auto t2 = _mm_unpacklo_ps(rows[2], rows[3]); // c0 d0 c1 d1
    auto t3 = _mm_unpackhi_ps(rows[2], rows[3]); // c2 d2 c3 d3
    rows[0] = _mm_movelh_ps(t0, t2);             // a0 b0 c0 d0
    rows[1] = _mm_movehl_ps(t2, t0);             // a1 b1 c1 d1
    rows[2] = _mm_movelh_ps(t1, t3);             // a2 b2 c2 d2
    rows[3] = _mm_movehl_ps(t3, t1);             // a3 b3 c3 d3
}

// ---------------------------------------------------------------------------
// ARM NEON (aarch64, including Apple Silicon): 128-bit registers
// ---------------------------------------------------------------------------
#elif defined(__aarch64__) || defined(_M_ARM64)

// 2×2 double transpose (NEON)
EINSUMS_FORCEINLINE void transpose_inplace(Vec<double> *rows) {
    auto lo = vzip1q_f64(rows[0], rows[1]);
    auto hi = vzip2q_f64(rows[0], rows[1]);
    rows[0] = lo;
    rows[1] = hi;
}

// 4×4 float transpose (NEON)
EINSUMS_FORCEINLINE void transpose_inplace(Vec<float> *rows) {
    float32x4x2_t t0 = vuzpq_f32(rows[0], rows[2]);
    float32x4x2_t t1 = vuzpq_f32(rows[1], rows[3]);
    float32x4x2_t t2 = vtrnq_f32(t0.val[0], t1.val[0]);
    float32x4x2_t t3 = vtrnq_f32(t0.val[1], t1.val[1]);
    rows[0]          = t2.val[0];
    rows[1]          = t3.val[0];
    rows[2]          = t2.val[1];
    rows[3]          = t3.val[1];
}

// 8×8 NEON 16-bit transpose helper.
//
// Three TRN stages (each across 8 vector ops): TRN1/TRN2 at u16 width
// transposes 2×2 sub-blocks, then at u32 width 4×4, then at u64 width 8×8.
// The output rows arrive permuted (0,4,2,6,1,5,3,7) which we resolve at
// the call site. Total: 24 vector ops, no memory round-trip.
namespace detail {
EINSUMS_FORCEINLINE void transpose_8x8_u16(uint16x8_t *r) {
    uint16x8_t const t0 = vtrn1q_u16(r[0], r[1]);
    uint16x8_t const t1 = vtrn2q_u16(r[0], r[1]);
    uint16x8_t const t2 = vtrn1q_u16(r[2], r[3]);
    uint16x8_t const t3 = vtrn2q_u16(r[2], r[3]);
    uint16x8_t const t4 = vtrn1q_u16(r[4], r[5]);
    uint16x8_t const t5 = vtrn2q_u16(r[4], r[5]);
    uint16x8_t const t6 = vtrn1q_u16(r[6], r[7]);
    uint16x8_t const t7 = vtrn2q_u16(r[6], r[7]);

    uint32x4_t const u0 = vtrn1q_u32(vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t2));
    uint32x4_t const u1 = vtrn2q_u32(vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t2));
    uint32x4_t const u2 = vtrn1q_u32(vreinterpretq_u32_u16(t1), vreinterpretq_u32_u16(t3));
    uint32x4_t const u3 = vtrn2q_u32(vreinterpretq_u32_u16(t1), vreinterpretq_u32_u16(t3));
    uint32x4_t const u4 = vtrn1q_u32(vreinterpretq_u32_u16(t4), vreinterpretq_u32_u16(t6));
    uint32x4_t const u5 = vtrn2q_u32(vreinterpretq_u32_u16(t4), vreinterpretq_u32_u16(t6));
    uint32x4_t const u6 = vtrn1q_u32(vreinterpretq_u32_u16(t5), vreinterpretq_u32_u16(t7));
    uint32x4_t const u7 = vtrn2q_u32(vreinterpretq_u32_u16(t5), vreinterpretq_u32_u16(t7));

    // Stage 3: u64 TRN gives the final rows, but in shuffled order.
    // After this step, results land at (0,4,2,6,1,5,3,7), so write back at the right slots.
    r[0] = vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u0), vreinterpretq_u64_u32(u4)));
    r[4] = vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u0), vreinterpretq_u64_u32(u4)));
    r[2] = vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u1), vreinterpretq_u64_u32(u5)));
    r[6] = vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u1), vreinterpretq_u64_u32(u5)));
    r[1] = vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u2), vreinterpretq_u64_u32(u6)));
    r[5] = vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u2), vreinterpretq_u64_u32(u6)));
    r[3] = vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u3), vreinterpretq_u64_u32(u7)));
    r[7] = vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u3), vreinterpretq_u64_u32(u7)));
}
} // namespace detail

#    if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
EINSUMS_FORCEINLINE void transpose_inplace(Vec<half_t> *rows) {
    constexpr int N = Vec<half_t>::lanes; // 8
    uint16x8_t    u[N];                   // NOLINT
    for (int i = 0; i < N; ++i)
        u[i] = vreinterpretq_u16_f16(rows[i].reg);
    detail::transpose_8x8_u16(u);
    for (int i = 0; i < N; ++i)
        rows[i].reg = vreinterpretq_f16_u16(u[i]);
}
#    endif

#    if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
EINSUMS_FORCEINLINE void transpose_inplace(Vec<bfloat16_t> *rows) {
    constexpr int N = Vec<bfloat16_t>::lanes; // 8
    uint16x8_t    u[N];                       // NOLINT
    for (int i = 0; i < N; ++i)
        u[i] = vreinterpretq_u16_bf16(rows[i].reg);
    detail::transpose_8x8_u16(u);
    for (int i = 0; i < N; ++i)
        rows[i].reg = vreinterpretq_bf16_u16(u[i]);
}
#    endif

// ---------------------------------------------------------------------------
// Scalar fallback: 1×1 is a no-op
// ---------------------------------------------------------------------------
#else

EINSUMS_FORCEINLINE void transpose_inplace(Vec<float> * /*rows*/) {
    // 1×1: nothing to do
}
EINSUMS_FORCEINLINE void transpose_inplace(Vec<double> * /*rows*/) {
    // 1×1: nothing to do
}

#endif

// ===========================================================================
// Complex transpose: transpose_inplace for CVec<T>
//
// Each complex<T> occupies 2*sizeof(T) bytes. So:
//   complex<float> on AVX: 4 complex values in 256 bits → 4×4 transpose
//     (same register pattern as double 4×4)
//   complex<double> on AVX: 2 complex values in 256 bits → 2×2 transpose
//     (same register pattern as double 2×2 on SSE2)
//   complex<float> on SSE2/NEON: 2 complex values → 2×2 transpose
//   complex<double> on SSE2/NEON: 1 complex value → no-op
//
// We reinterpret CVec<T> registers and delegate to the double/float
// transpose at the appropriate width.
// ===========================================================================

/// Transpose N×N block of complex<float> values in-place.
/// N = CVec<float>::complex_lanes.
EINSUMS_FORCEINLINE void complex_transpose_inplace(CVec<float> *rows) {
    constexpr int N = CVec<float>::complex_lanes;
    if constexpr (N <= 1) {
        // 1×1 or 0: nothing to do
        return;
    }

    // Each complex<float> is 64 bits (same as double).
    // Reinterpret the Vec<float> registers as if they hold N "double-width"
    // elements and use the store-transpose-load approach.
    alignas(native_alignment) float buf[N * N * 2]; // N rows × N complex values × 2 floats
    for (int i = 0; i < N; ++i)
        storeu(&buf[i * N * 2], rows[i].reg);

    // Scalar transpose of complex pairs
    // buf is N×N complex matrix in row-major, each complex = 2 floats
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            // Swap complex element (i,j) with (j,i)
            std::swap(buf[(i * N + j) * 2], buf[(j * N + i) * 2]);
            std::swap(buf[(i * N + j) * 2 + 1], buf[(j * N + i) * 2 + 1]);
        }
    }

    for (int i = 0; i < N; ++i)
        rows[i].reg = loadu(&buf[i * N * 2]);
}

/// Transpose N×N block of complex<double> values in-place.
/// N = CVec<double>::complex_lanes.
EINSUMS_FORCEINLINE void complex_transpose_inplace(CVec<double> *rows) {
    constexpr int N = CVec<double>::complex_lanes;
    if constexpr (N <= 1) {
        return;
    }

    alignas(native_alignment) double buf[N * N * 2];
    for (int i = 0; i < N; ++i)
        storeu(&buf[i * N * 2], rows[i].reg);

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            std::swap(buf[(i * N + j) * 2], buf[(j * N + i) * 2]);
            std::swap(buf[(i * N + j) * 2 + 1], buf[(j * N + i) * 2 + 1]);
        }
    }

    for (int i = 0; i < N; ++i)
        rows[i].reg = loadu(&buf[i * N * 2]);
}

} // namespace einsums::simd
