//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <complex>

namespace einsums::simd {

// ===========================================================================
// CVec<T>: interleaved complex SIMD vector.
//
// Wraps a Vec<T> register containing N/2 complex<T> values stored as
// [re0, im0, re1, im1, ...]. This matches the memory layout of
// std::complex<T>.
//
// Type T is the underlying real type (float or double).
// ===========================================================================

template <typename T>
struct CVec {
    static_assert(std::is_floating_point_v<T>, "CVec<T> requires float or double");

    Vec<T> reg;

    /// Number of complex values held (half the real lanes)
    static constexpr int complex_lanes = Vec<T>::lanes / 2;
    static constexpr int real_lanes    = Vec<T>::lanes;

    CVec() = default;
    EINSUMS_FORCEINLINE CVec(Vec<T> r) : reg(r) {}
    EINSUMS_FORCEINLINE CVec(typename Vec<T>::reg_type r) : reg(r) {}
    EINSUMS_FORCEINLINE operator Vec<T>() const { return reg; }
    EINSUMS_FORCEINLINE operator typename Vec<T>::reg_type() const { return reg.reg; }
};

// ===========================================================================
// Load / Store: cast complex<T>* to T* and use real load/store
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE CVec<T> complex_loadu(std::complex<T> const *ptr) {
    return loadu(reinterpret_cast<T const *>(ptr));
}

template <typename T>
EINSUMS_FORCEINLINE void complex_storeu(std::complex<T> *ptr, CVec<T> v) {
    storeu(reinterpret_cast<T *>(ptr), v.reg);
}

// ===========================================================================
// Broadcast: complex scalar → CVec<T>
// Fills register with [re, im, re, im, ...]
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE CVec<T> complex_broadcast(std::complex<T> val);

#if defined(__AVX512F__)
template <>
EINSUMS_FORCEINLINE CVec<float> complex_broadcast(std::complex<float> val) {
    return _mm512_setr_ps(val.real(), val.imag(), val.real(), val.imag(), val.real(), val.imag(), val.real(), val.imag(), val.real(),
                          val.imag(), val.real(), val.imag(), val.real(), val.imag(), val.real(), val.imag());
}
template <>
EINSUMS_FORCEINLINE CVec<double> complex_broadcast(std::complex<double> val) {
    return _mm512_setr_pd(val.real(), val.imag(), val.real(), val.imag(), val.real(), val.imag(), val.real(), val.imag());
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE CVec<float> complex_broadcast(std::complex<float> val) {
    return _mm256_setr_ps(val.real(), val.imag(), val.real(), val.imag(), val.real(), val.imag(), val.real(), val.imag());
}
template <>
EINSUMS_FORCEINLINE CVec<double> complex_broadcast(std::complex<double> val) {
    return _mm256_setr_pd(val.real(), val.imag(), val.real(), val.imag());
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE CVec<float> complex_broadcast(std::complex<float> val) {
    return _mm_setr_ps(val.real(), val.imag(), val.real(), val.imag());
}
template <>
EINSUMS_FORCEINLINE CVec<double> complex_broadcast(std::complex<double> val) {
    return _mm_setr_pd(val.real(), val.imag());
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE CVec<float> complex_broadcast(std::complex<float> val) {
    float tmp[4] = {val.real(), val.imag(), val.real(), val.imag()};
    return vld1q_f32(tmp);
}
template <>
EINSUMS_FORCEINLINE CVec<double> complex_broadcast(std::complex<double> val) {
    double tmp[2] = {val.real(), val.imag()};
    return vld1q_f64(tmp);
}
#else
template <>
EINSUMS_FORCEINLINE CVec<float> complex_broadcast(std::complex<float>) {
    return {0.0f}; // scalar fallback: CVec holds 0 complex values
}
template <>
EINSUMS_FORCEINLINE CVec<double> complex_broadcast(std::complex<double>) {
    return {0.0};
}
#endif

// ===========================================================================
// Add / Sub: element-wise, identical to real operations
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE CVec<T> complex_add(CVec<T> a, CVec<T> b) {
    return add(a.reg, b.reg);
}

template <typename T>
EINSUMS_FORCEINLINE CVec<T> complex_sub(CVec<T> a, CVec<T> b) {
    return sub(a.reg, b.reg);
}

// ===========================================================================
// Conjugate: negate imaginary parts (every odd element)
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE CVec<T> conjugate(CVec<T> v);

#if defined(__AVX512F__)
template <>
EINSUMS_FORCEINLINE CVec<float> conjugate(CVec<float> v) {
    auto sign = _mm512_setr_ps(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f);
    return _mm512_xor_ps(v.reg, sign);
}
template <>
EINSUMS_FORCEINLINE CVec<double> conjugate(CVec<double> v) {
    auto sign = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    return _mm512_xor_pd(v.reg, sign);
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE CVec<float> conjugate(CVec<float> v) {
    auto sign = _mm256_setr_ps(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f);
    return _mm256_xor_ps(v.reg, sign);
}
template <>
EINSUMS_FORCEINLINE CVec<double> conjugate(CVec<double> v) {
    auto sign = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
    return _mm256_xor_pd(v.reg, sign);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE CVec<float> conjugate(CVec<float> v) {
    auto sign = _mm_setr_ps(0.f, -0.f, 0.f, -0.f);
    return _mm_xor_ps(v.reg, sign);
}
template <>
EINSUMS_FORCEINLINE CVec<double> conjugate(CVec<double> v) {
    auto sign = _mm_setr_pd(0.0, -0.0);
    return _mm_xor_pd(v.reg, sign);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE CVec<float> conjugate(CVec<float> v) {
    // Negate odd lanes (imaginary parts) via XOR with sign bit
    static float const conj_data[4] = {0.f, -0.f, 0.f, -0.f};
    auto               sign         = vld1q_f32(conj_data);
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v.reg), vreinterpretq_u32_f32(sign)));
}
template <>
EINSUMS_FORCEINLINE CVec<double> conjugate(CVec<double> v) {
    static double const conj_data[2] = {0.0, -0.0};
    auto                sign         = vld1q_f64(conj_data);
    return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(v.reg), vreinterpretq_u64_f64(sign)));
}
#else
template <>
EINSUMS_FORCEINLINE CVec<float> conjugate(CVec<float> v) {
    return v;
}
template <>
EINSUMS_FORCEINLINE CVec<double> conjugate(CVec<double> v) {
    return v;
}
#endif

// ===========================================================================
// Complex multiply: (a_re + i*a_im) * (b_re + i*b_im)
//   result_re = a_re*b_re - a_im*b_im
//   result_im = a_re*b_im + a_im*b_re
//
// For interleaved [re,im,re,im,...]:
//   1. Broadcast real parts of a: a_rr = [re,re,re,re,...]
//   2. Broadcast imag parts of a: a_ii = [im,im,im,im,...]
//   3. t1 = a_rr * b                    = [re*bre, re*bim, ...]
//   4. b_swap = swap re<->im in b       = [bim, bre, bim, bre, ...]
//   5. t2 = a_ii * b_swap               = [im*bim, im*bre, ...]
//   6. result = addsub(t1, t2)          = [re*bre - im*bim, re*bim + im*bre]
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE CVec<T> complex_mul(CVec<T> a, CVec<T> b);

#if defined(__AVX512F__)
template <>
EINSUMS_FORCEINLINE CVec<float> complex_mul(CVec<float> a, CVec<float> b) {
    auto a_rr   = _mm512_moveldup_ps(a.reg);
    auto a_ii   = _mm512_movehdup_ps(a.reg);
    auto b_swap = _mm512_shuffle_ps(b.reg, b.reg, 0xB1);
    auto t1     = _mm512_mul_ps(a_rr, b.reg);
    auto t2     = _mm512_mul_ps(a_ii, b_swap);
    // AVX-512 has no addsub; use fmaddsub: a*b ± c  →  fmaddsub(a_rr, b, -t2) won't work directly.
    // Instead: negate even lanes of t2, then add.
    auto neg_mask = _mm512_setr_ps(-1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f);
    return _mm512_add_ps(t1, _mm512_mul_ps(t2, neg_mask));
}
template <>
EINSUMS_FORCEINLINE CVec<double> complex_mul(CVec<double> a, CVec<double> b) {
    auto a_rr     = _mm512_movedup_pd(a.reg);
    auto a_ii     = _mm512_shuffle_pd(a.reg, a.reg, 0xFF);
    auto b_swap   = _mm512_shuffle_pd(b.reg, b.reg, 0x55);
    auto t1       = _mm512_mul_pd(a_rr, b.reg);
    auto t2       = _mm512_mul_pd(a_ii, b_swap);
    auto neg_mask = _mm512_setr_pd(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    return _mm512_add_pd(t1, _mm512_mul_pd(t2, neg_mask));
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE CVec<float> complex_mul(CVec<float> a, CVec<float> b) {
    // Duplicate real parts: [r0,r0,r1,r1,...] and imag parts: [i0,i0,i1,i1,...]
    auto a_rr = _mm256_moveldup_ps(a.reg); // broadcast even lanes (real)
    auto a_ii = _mm256_movehdup_ps(a.reg); // broadcast odd lanes (imag)
    // Swap re<->im in b
    auto b_swap = _mm256_shuffle_ps(b.reg, b.reg, 0xB1); // 10_11_00_01
    auto t1     = _mm256_mul_ps(a_rr, b.reg);
    auto t2     = _mm256_mul_ps(a_ii, b_swap);
    return _mm256_addsub_ps(t1, t2); // even: t1-t2, odd: t1+t2
}
template <>
EINSUMS_FORCEINLINE CVec<double> complex_mul(CVec<double> a, CVec<double> b) {
    auto a_rr   = _mm256_movedup_pd(a.reg);             // [r0,r0,r1,r1]
    auto a_ii   = _mm256_shuffle_pd(a.reg, a.reg, 0xF); // [i0,i0,i1,i1]
    auto b_swap = _mm256_shuffle_pd(b.reg, b.reg, 0x5); // swap re<->im
    auto t1     = _mm256_mul_pd(a_rr, b.reg);
    auto t2     = _mm256_mul_pd(a_ii, b_swap);
    return _mm256_addsub_pd(t1, t2);
}
#elif defined(__SSE3__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
// SSE3+ path: moveldup/movehdup/movedup and addsub are SSE3 intrinsics. MSVC always exposes
// these regardless of the target feature level, so the MSVC checks stay here.
template <>
EINSUMS_FORCEINLINE CVec<float> complex_mul(CVec<float> a, CVec<float> b) {
    auto a_rr   = _mm_moveldup_ps(a.reg);
    auto a_ii   = _mm_movehdup_ps(a.reg);
    auto b_swap = _mm_shuffle_ps(b.reg, b.reg, 0xB1);
    auto t1     = _mm_mul_ps(a_rr, b.reg);
    auto t2     = _mm_mul_ps(a_ii, b_swap);
    return _mm_addsub_ps(t1, t2);
}
template <>
EINSUMS_FORCEINLINE CVec<double> complex_mul(CVec<double> a, CVec<double> b) {
    auto a_rr   = _mm_movedup_pd(a.reg);
    auto a_ii   = _mm_shuffle_pd(a.reg, a.reg, 0x3);
    auto b_swap = _mm_shuffle_pd(b.reg, b.reg, 0x1);
    auto t1     = _mm_mul_pd(a_rr, b.reg);
    auto t2     = _mm_mul_pd(a_ii, b_swap);
    return _mm_addsub_pd(t1, t2);
}
#elif defined(__SSE2__)
// SSE2-only fallback (e.g. the generic x86-64 baseline used by the conda toolchain): no
// moveldup/movehdup/movedup or addsub. Duplicate the real/imag lanes with plain shuffles and
// emulate addsub by multiplying the second product by an alternating [-1,+1,...] mask.
template <>
EINSUMS_FORCEINLINE CVec<float> complex_mul(CVec<float> a, CVec<float> b) {
    auto a_rr   = _mm_shuffle_ps(a.reg, a.reg, 0xA0); // [r0,r0,r1,r1]
    auto a_ii   = _mm_shuffle_ps(a.reg, a.reg, 0xF5); // [i0,i0,i1,i1]
    auto b_swap = _mm_shuffle_ps(b.reg, b.reg, 0xB1); // [i,r,i,r]
    auto t1     = _mm_mul_ps(a_rr, b.reg);
    auto t2     = _mm_mul_ps(a_ii, b_swap);
    auto neg    = _mm_set_ps(1.f, -1.f, 1.f, -1.f); // lanes [-1,+1,-1,+1]
    return _mm_add_ps(t1, _mm_mul_ps(t2, neg));     // even: t1-t2, odd: t1+t2
}
template <>
EINSUMS_FORCEINLINE CVec<double> complex_mul(CVec<double> a, CVec<double> b) {
    auto a_rr   = _mm_shuffle_pd(a.reg, a.reg, 0x0); // [r0,r0]
    auto a_ii   = _mm_shuffle_pd(a.reg, a.reg, 0x3); // [i0,i0]
    auto b_swap = _mm_shuffle_pd(b.reg, b.reg, 0x1); // [i0,r0]
    auto t1     = _mm_mul_pd(a_rr, b.reg);
    auto t2     = _mm_mul_pd(a_ii, b_swap);
    auto neg    = _mm_set_pd(1.0, -1.0); // lanes [-1,+1]
    return _mm_add_pd(t1, _mm_mul_pd(t2, neg));
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE CVec<float> complex_mul(CVec<float> a, CVec<float> b) {
    // NEON FCMLA (complex multiply-accumulate) is available on ARMv8.3+
    // For portability, use the explicit pattern:
    // a_rr = [a0r, a0r, a1r, a1r], a_ii = [a0i, a0i, a1i, a1i]
    auto a_rr   = vtrn1q_f32(a.reg, a.reg); // duplicate even lanes
    auto a_ii   = vtrn2q_f32(a.reg, a.reg); // duplicate odd lanes
    auto b_swap = vrev64q_f32(b.reg);       // swap pairs within 64-bit lanes
    auto t1     = vmulq_f32(a_rr, b.reg);   // [ar*br, ar*bi, ...]
    auto t2     = vmulq_f32(a_ii, b_swap);  // [ai*bi, ai*br, ...]
    // addsub: even lanes subtract, odd lanes add
    static float const negate_data[4] = {-1.0f, 1.0f, -1.0f, 1.0f};
    auto               neg            = vld1q_f32(negate_data);
    return vaddq_f32(t1, vmulq_f32(t2, neg)); // t1[0]-t2[0], t1[1]+t2[1], ...
}
template <>
EINSUMS_FORCEINLINE CVec<double> complex_mul(CVec<double> a, CVec<double> b) {
    auto a_rr = vtrn1q_f64(a.reg, a.reg);
    auto a_ii = vtrn2q_f64(a.reg, a.reg);
    // Swap re<->im: for 2 doubles, just use ext
    auto                b_swap         = vextq_f64(b.reg, b.reg, 1);
    auto                t1             = vmulq_f64(a_rr, b.reg);
    auto                t2             = vmulq_f64(a_ii, b_swap);
    static double const negate_data[2] = {-1.0, 1.0};
    auto                neg            = vld1q_f64(negate_data);
    return vaddq_f64(t1, vmulq_f64(t2, neg));
}
#else
template <>
EINSUMS_FORCEINLINE CVec<float> complex_mul(CVec<float> a, CVec<float> b) {
    return a; // scalar: single element, not meaningful
}
template <>
EINSUMS_FORCEINLINE CVec<double> complex_mul(CVec<double> a, CVec<double> b) {
    return a;
}
#endif

// ===========================================================================
// Complex FMA: a * b + c  (all complex)
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE CVec<T> complex_fmadd(CVec<T> a, CVec<T> b, CVec<T> c) {
    return complex_add(complex_mul(a, b), c);
}

// ===========================================================================
// Complex scale: CVec * real scalar (broadcast alpha.real to all lanes)
// This is for the common case of scaling by a real-valued alpha.
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE CVec<T> complex_scale(CVec<T> v, T scalar) {
    return mul(v.reg, broadcast(scalar));
}

// ===========================================================================
// Complex gather/scatter: gather complex<T> values from strided memory
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE CVec<T> complex_gather(std::complex<T> const *base, std::ptrdiff_t stride) {
    // Each complex<T> is 2*sizeof(T) bytes. Stride is in units of complex<T>.
    // We need to load complex_lanes values from base[0], base[stride], base[2*stride], ...
    constexpr int N = CVec<T>::complex_lanes;
    if (stride == 1) {
        return complex_loadu(base);
    }
    // Scalar gather for complex values
    alignas(native_alignment) T buf[N * 2];
    for (int i = 0; i < N; ++i) {
        buf[2 * i]     = base[i * stride].real();
        buf[2 * i + 1] = base[i * stride].imag();
    }
    return loada(buf);
}

template <typename T>
EINSUMS_FORCEINLINE void complex_scatter(std::complex<T> *base, std::ptrdiff_t stride, CVec<T> v) {
    constexpr int N = CVec<T>::complex_lanes;
    if (stride == 1) {
        complex_storeu(base, v);
        return;
    }
    // Scalar scatter for complex values
    alignas(native_alignment) T buf[N * 2];
    storea(buf, v.reg);
    for (int i = 0; i < N; ++i) {
        base[i * stride] = std::complex<T>(buf[2 * i], buf[2 * i + 1]);
    }
}

// ===========================================================================
// Operator overloads
// ===========================================================================

#if !defined(EINSUMS_SIMD_NO_OPERATORS)
template <typename T>
EINSUMS_FORCEINLINE CVec<T> operator+(CVec<T> a, CVec<T> b) {
    return complex_add(a, b);
}
template <typename T>
EINSUMS_FORCEINLINE CVec<T> operator-(CVec<T> a, CVec<T> b) {
    return complex_sub(a, b);
}
template <typename T>
EINSUMS_FORCEINLINE CVec<T> operator*(CVec<T> a, CVec<T> b) {
    return complex_mul(a, b);
}
#endif

} // namespace einsums::simd
