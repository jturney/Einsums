//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/SIMD/Convert.hpp>
#include <Einsums/SIMD/Gather.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Partial.hpp>
#include <Einsums/SIMD/Platform.hpp>
#include <Einsums/SIMD/Vec.hpp>
#include <Einsums/SIMD/Wide.hpp>

#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>

// ===========================================================================
// One kernel body for vectors and scalars.
//
// A kernel written as a template over its value type V instantiates with
// V = Vec<double> on the CPU, a lane per problem, and with V = double for one
// problem at a time: on a GPU thread, or on the CPU as a reference. Every
// operation it calls therefore needs one spelling for both. This header adds
// the scalar half of the operations in Operations.hpp, and the few generic
// operations that have no vector spelling to reuse:
//
//   scalar_t<V>, lanes_v<V>, is_vec_v<V>   element type, lane count, and whether V is a Vec<T, N>
//   splat<V>(x)                           broadcast(x), or x
//   load<V>(p), store(p, v)               loadu and storeu, or *p
//   lookup(base, idx)                     gather(base, idx) with a lane per index, or base[idx]
//
// The scalar overloads match one lane of the vector operation bit for bit, so
// a scalar run on the CPU is an exact reference for the vector run, as long as
// the compiler does not contract a kernel's own separate multiply and add into
// an FMA: GCC does by default (-ffp-contract=fast), independently in each
// instantiation, so build the comparison with -ffp-contract=off. In particular: min is
// a < b ? a : b (not std::fmin), the fused forms round once exactly where the
// vector forms do (where the build has FMA), and a compare returns bool where
// the vector one returns a Mask<T>. Everything that takes a Mask<T> takes the
// bool too: select, any, all, none, count, the bitwise mask combinations, and
// the masked loadu, storeu and lookup. Write !m, not ~m, in generic code.
//
// Call them qualified, einsums::simd::fmadd(...): argument-dependent lookup
// finds the vector forms but nothing for double. Each scalar overload is a
// template constrained to arithmetic types, never a plain function, so under
// using namespace einsums::simd an unqualified sqrt(2.0) still means the C
// library's sqrt. std::min and std::max are templates too, so code that has
// both using namespace std and using namespace einsums::simd must qualify an
// unqualified min or max on scalars, which would otherwise be ambiguous.
// ===========================================================================

EINSUMS_NAMESPACE_BEGIN(simd)
EINSUMS_SIMD_ISA_NAMESPACE_BEGIN()

// ---------------------------------------------------------------------------
// Traits.
// ---------------------------------------------------------------------------

namespace detail {
template <typename V>
struct is_vec : std::false_type {};
template <typename T, int N>
struct is_vec<Vec<T, N>> : std::true_type {};

template <typename V>
struct lanes_of : std::integral_constant<int, 1> {};
template <typename T, int N>
struct lanes_of<Vec<T, N>> : std::integral_constant<int, N> {};

template <typename V>
struct value_of {
    using type = V;
};
template <typename T, int N>
struct value_of<Vec<T, N>> {
    using type = T;
};

/// An element type a scalar overload takes: a floating-point type or an integer, but not bool.
template <typename T>
concept arithmetic = (std::floating_point<T> || std::integral<T>) && !std::same_as<T, bool>;
} // namespace detail

/// Whether V is a Vec<T>.
template <typename V>
inline constexpr bool is_vec_v = detail::is_vec<V>::value;

/// The element type of V: T for both Vec<T> and T.
template <typename V>
using scalar_t = typename detail::value_of<V>::type;

/// The number of problems one V holds: Vec<T>::lanes, or 1 for a scalar.
template <typename V>
inline constexpr int lanes_v = detail::lanes_of<V>::value;

// ---------------------------------------------------------------------------
// Construction, load, store and table lookup.
// ---------------------------------------------------------------------------

/// A V holding x in every lane.
template <typename V>
EINSUMS_FORCEINLINE V splat(scalar_t<V> x) {
    if constexpr (!is_vec_v<V>) {
        return x;
    } else if constexpr (V::native) {
        return broadcast(x);
    } else {
        V r;
        r.part.fill(broadcast(x));
        return r;
    }
}

/// lanes_v<V> consecutive elements from p, which need no alignment.
template <typename V>
EINSUMS_FORCEINLINE V load(scalar_t<V> const *p) {
    if constexpr (!is_vec_v<V>) {
        return *p;
    } else if constexpr (V::native) {
        return loadu(p);
    } else {
        constexpr int L = V::part_type::lanes;
        V             r;
        for (int k = 0; k < V::parts; ++k) {
            r.part[k] = loadu(p + k * L);
        }
        return r;
    }
}

/// Write v's lanes to consecutive elements from p, which needs no alignment.
template <typename T, int N>
EINSUMS_FORCEINLINE void store(T *p, Vec<T, N> v) {
    storeu(p, v);
}
template <detail::arithmetic T>
EINSUMS_FORCEINLINE void store(T *p, T v) {
    *p = v;
}

/// base[idx] for each lane's index: the index gather for a vector, plain indexing for a scalar. A
/// distinct name from gather, whose (base, integer) form is a strided load and would silently
/// accept a scalar index.
template <typename T>
EINSUMS_FORCEINLINE Vec<T> lookup(T const *base, Vec<gather_index_t<T>> idx) {
    return gather(base, idx);
}
template <std::floating_point T, std::integral I>
EINSUMS_FORCEINLINE T lookup(T const *base, I idx) {
    return base[idx];
}
/// lookup in the lanes m sets, zero elsewhere; an inactive lane's index is never dereferenced.
template <typename T>
EINSUMS_FORCEINLINE Vec<T> lookup(T const *base, Vec<gather_index_t<T>> idx, Mask<T> m) {
    return gather(base, idx, m);
}
template <std::floating_point T, std::integral I>
EINSUMS_FORCEINLINE T lookup(T const *base, I idx, bool m) {
    return m ? base[idx] : T(0);
}

// ---------------------------------------------------------------------------
// lookup on wide vectors, with an index of either width and the same lane count.
//
// A Vec<int32_t, N> index lets the two tiers of a mixed-precision kernel share
// one index vector: the FP32 tier computes convert<int32_t>(floor(x)) and the
// FP64 tier, Vec<double, N>, gathers its doubles with the same register. AVX2
// and AVX-512 gather doubles at 32-bit indices directly, half an index register
// per double register; SSE takes the indices out of the register and loads the
// doubles one by one; every other build reads lane by lane through memory.
// ---------------------------------------------------------------------------

namespace detail {
template <typename I>
concept lookup_index = std::same_as<I, int32_t> || std::same_as<I, int64_t>;

/// Whether lookup(base, Vec<I, N>) is the native index gather of gather_index_t<T>, which the plain
/// overloads above already are.
template <typename T, typename I, int N>
inline constexpr bool native_lookup = N == VecTraits<T>::lanes && std::same_as<I, gather_index_t<T>>;

#if (defined(__AVX512F__) && defined(__AVX512VL__)) || defined(__AVX2__)
#    define EINSUMS_SIMD_HAVE_GATHER_PD_I32 1
/// The doubles at half @p h of a 32-bit index register: one double register's worth of lanes.
EINSUMS_FORCEINLINE Vec<double> gather_pd_i32(double const *base, Vec<int32_t> idx, int h) {
#    if defined(__AVX512F__) && defined(__AVX512VL__)
    __m256i const half = h == 0 ? _mm512_castsi512_si256(idx.reg) : _mm512_extracti64x4_epi64(idx.reg, 1);
    return _mm512_i32gather_pd(half, base, sizeof(double));
#    else
    __m128i const half = h == 0 ? _mm256_castsi256_si128(idx.reg) : _mm256_extracti128_si256(idx.reg, 1);
    return _mm256_i32gather_pd(base, half, sizeof(double));
#    endif
}
EINSUMS_FORCEINLINE Vec<double> gather_pd_i32(double const *base, Vec<int32_t> idx, int h, Mask<double> m) {
#    if defined(__AVX512F__) && defined(__AVX512VL__)
    __m256i const half = h == 0 ? _mm512_castsi512_si256(idx.reg) : _mm512_extracti64x4_epi64(idx.reg, 1);
    return _mm512_mask_i32gather_pd(_mm512_setzero_pd(), m.reg, half, base, sizeof(double));
#    else
    __m128i const half = h == 0 ? _mm256_castsi256_si128(idx.reg) : _mm256_extracti128_si256(idx.reg, 1);
    return _mm256_mask_i32gather_pd(_mm256_setzero_pd(), base, half, m.reg, sizeof(double));
#    endif
}
#elif !defined(__AVX__) && (defined(__x86_64__) || defined(_M_X64))
#    define EINSUMS_SIMD_HAVE_GATHER_PD_I32 1
// SSE has no gather: the two indices of half h come out of the register and the doubles are loaded
// into one, with no round trip through memory, which stalls on store forwarding.
EINSUMS_FORCEINLINE Vec<double> gather_pd_i32(double const *base, Vec<int32_t> idx, int h) {
    __m128i const pair = h == 0 ? idx.reg : _mm_unpackhi_epi64(idx.reg, idx.reg);
    int const     i0   = _mm_cvtsi128_si32(pair);
    int const     i1   = _mm_cvtsi128_si32(_mm_shuffle_epi32(pair, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_setr_pd(base[i0], base[i1]);
}
EINSUMS_FORCEINLINE Vec<double> gather_pd_i32(double const *base, Vec<int32_t> idx, int h, Mask<double> m) {
    __m128i const pair = h == 0 ? idx.reg : _mm_unpackhi_epi64(idx.reg, idx.reg);
    int const     set  = _mm_movemask_pd(m.reg);
    double const  d0   = (set & 1) ? base[_mm_cvtsi128_si32(pair)] : 0.0;
    double const  d1   = (set & 2) ? base[_mm_cvtsi128_si32(_mm_shuffle_epi32(pair, _MM_SHUFFLE(1, 1, 1, 1)))] : 0.0;
    return _mm_setr_pd(d0, d1);
}
#endif

/// The fallback: every lane's index read through memory, and only the lanes @p bits sets loaded.
template <typename T, typename I, int N>
EINSUMS_FORCEINLINE Vec<T, N> lookup_lanes(T const *base, Vec<I, N> idx, uint64_t bits) {
    alignas(native_alignment) I at[N];
    alignas(native_alignment) T out[N] = {};
    store(at, idx);
    for (int i = 0; i < N; ++i) {
        if ((bits >> i) & 1u) {
            out[i] = base[at[i]];
        }
    }
    return load<Vec<T, N>>(out);
}
} // namespace detail

template <std::floating_point T, detail::lookup_index I, int N>
    requires(!detail::native_lookup<T, I, N>)
EINSUMS_FORCEINLINE Vec<T, N> lookup(T const *base, Vec<I, N> idx) {
    constexpr int LT = VecTraits<T>::lanes;
    constexpr int LI = VecTraits<I>::lanes;
    if constexpr (LT == LI && std::same_as<I, gather_index_t<T>>) {
        Vec<T, N> r;
        for (int k = 0; k < N / LT; ++k) {
            detail::set_part(r, k, gather(base, detail::part_of(idx, k)));
        }
        return r;
    }
#if defined(EINSUMS_SIMD_HAVE_GATHER_PD_I32)
    else if constexpr (std::same_as<T, double> && std::same_as<I, int32_t> && LI == 2 * LT) {
        Vec<T, N> r;
        for (int k = 0; k < N / LI; ++k) {
            Vec<int32_t> const p = detail::part_of(idx, k);
            detail::set_part(r, 2 * k, detail::gather_pd_i32(base, p, 0));
            detail::set_part(r, 2 * k + 1, detail::gather_pd_i32(base, p, 1));
        }
        return r;
    }
#endif
    else {
        return detail::lookup_lanes(base, idx, N == 64 ? ~uint64_t{0} : (uint64_t{1} << N) - 1u);
    }
}

template <std::floating_point T, detail::lookup_index I, int N>
    requires(!detail::native_lookup<T, I, N>)
EINSUMS_FORCEINLINE Vec<T, N> lookup(T const *base, Vec<I, N> idx, Mask<T, N> m) {
    constexpr int LT = VecTraits<T>::lanes;
    constexpr int LI = VecTraits<I>::lanes;
    if constexpr (LT == LI && std::same_as<I, gather_index_t<T>>) {
        Vec<T, N> r;
        for (int k = 0; k < N / LT; ++k) {
            detail::set_part(r, k, gather(base, detail::part_of(idx, k), detail::part_of(m, k)));
        }
        return r;
    }
#if defined(EINSUMS_SIMD_HAVE_GATHER_PD_I32)
    else if constexpr (std::same_as<T, double> && std::same_as<I, int32_t> && LI == 2 * LT) {
        Vec<T, N> r;
        for (int k = 0; k < N / LI; ++k) {
            Vec<int32_t> const p = detail::part_of(idx, k);
            detail::set_part(r, 2 * k, detail::gather_pd_i32(base, p, 0, detail::part_of(m, 2 * k)));
            detail::set_part(r, 2 * k + 1, detail::gather_pd_i32(base, p, 1, detail::part_of(m, 2 * k + 1)));
        }
        return r;
    }
#endif
    else {
        return detail::lookup_lanes(base, idx, to_bits(m));
    }
}

/// The scalar forms of the masked load and store: *p where m is true, and zero or nothing otherwise.
template <detail::arithmetic T>
EINSUMS_FORCEINLINE T loadu(T const *p, bool m) {
    return m ? *p : T(0);
}
template <detail::arithmetic T>
EINSUMS_FORCEINLINE void storeu(T *p, T v, bool m) {
    if (m) {
        *p = v;
    }
}

// ---------------------------------------------------------------------------
// Scalar arithmetic and math, matching one lane of the vector forms.
// ---------------------------------------------------------------------------

/// Whether the fused forms round once on this build, as the vector fmadd does.
inline constexpr bool scalar_fma_fused =
#if defined(EINSUMS_SIMD_HAVE_FMA) || defined(__aarch64__) || defined(_M_ARM64)
    true;
#else
    false;
#endif

template <std::floating_point T>
EINSUMS_FORCEINLINE T fmadd(T a, T b, T c) {
    if constexpr (scalar_fma_fused) {
        return std::fma(a, b, c);
    } else {
        return a * b + c;
    }
}
template <std::floating_point T>
EINSUMS_FORCEINLINE T fmsub(T a, T b, T c) {
    if constexpr (scalar_fma_fused) {
        return std::fma(a, b, -c);
    } else {
        return a * b - c;
    }
}
template <std::floating_point T>
EINSUMS_FORCEINLINE T fnmadd(T a, T b, T c) {
    if constexpr (scalar_fma_fused) {
        return std::fma(-a, b, c);
    } else {
        return c - a * b;
    }
}
template <std::floating_point T>
EINSUMS_FORCEINLINE T fnmsub(T a, T b, T c) {
    if constexpr (scalar_fma_fused) {
        return std::fma(-a, b, -c);
    } else {
        return -(a * b) - c;
    }
}

template <std::floating_point T>
EINSUMS_FORCEINLINE T div(T a, T b) {
    return a / b;
}
template <std::floating_point T>
EINSUMS_FORCEINLINE T sqrt(T a) {
    return std::sqrt(a);
}
/// a < b ? a : b exactly, as the vector min: a NaN on either side, or two zeros, give b.
template <std::floating_point T>
EINSUMS_FORCEINLINE T min(T a, T b) {
    return a < b ? a : b;
}
/// a > b ? a : b exactly, as the vector max.
template <std::floating_point T>
EINSUMS_FORCEINLINE T max(T a, T b) {
    return a > b ? a : b;
}
template <std::floating_point T>
EINSUMS_FORCEINLINE T abs(T a) {
    return std::fabs(a);
}
template <std::floating_point T>
EINSUMS_FORCEINLINE T neg(T a) {
    return -a;
}

template <std::floating_point T>
EINSUMS_FORCEINLINE T floor(T x) {
    return std::floor(x);
}
template <std::floating_point T>
EINSUMS_FORCEINLINE T ceil(T x) {
    return std::ceil(x);
}
template <std::floating_point T>
EINSUMS_FORCEINLINE T trunc(T x) {
    return std::trunc(x);
}
template <std::floating_point T>
EINSUMS_FORCEINLINE T round(T x) {
    return std::round(x);
}
/// Ties to even, in the default rounding mode; the vector form ignores the mode.
template <std::floating_point T>
EINSUMS_FORCEINLINE T round_even(T x) {
    return std::nearbyint(x);
}

// ---------------------------------------------------------------------------
// Scalar comparisons and masks: a compare gives a bool.
// ---------------------------------------------------------------------------

template <detail::arithmetic T>
EINSUMS_FORCEINLINE bool cmp_eq(T a, T b) {
    return a == b;
}
template <detail::arithmetic T>
EINSUMS_FORCEINLINE bool cmp_ne(T a, T b) {
    return a != b;
}
template <detail::arithmetic T>
EINSUMS_FORCEINLINE bool cmp_lt(T a, T b) {
    return a < b;
}
template <detail::arithmetic T>
EINSUMS_FORCEINLINE bool cmp_le(T a, T b) {
    return a <= b;
}
template <detail::arithmetic T>
EINSUMS_FORCEINLINE bool cmp_gt(T a, T b) {
    return a > b;
}
template <detail::arithmetic T>
EINSUMS_FORCEINLINE bool cmp_ge(T a, T b) {
    return a >= b;
}

template <detail::arithmetic T>
EINSUMS_FORCEINLINE T select(bool mask, T a, T b) {
    return mask ? a : b;
}
template <std::same_as<bool> B>
EINSUMS_FORCEINLINE bool any(B mask) {
    return mask;
}
template <std::same_as<bool> B>
EINSUMS_FORCEINLINE bool all(B mask) {
    return mask;
}
template <std::same_as<bool> B>
EINSUMS_FORCEINLINE bool none(B mask) {
    return !mask;
}
template <std::same_as<bool> B>
EINSUMS_FORCEINLINE int count(B mask) {
    return mask ? 1 : 0;
}
template <std::same_as<bool> B>
EINSUMS_FORCEINLINE bool bitwise_and(B a, B b) {
    return a && b;
}
template <std::same_as<bool> B>
EINSUMS_FORCEINLINE bool bitwise_or(B a, B b) {
    return a || b;
}
template <std::same_as<bool> B>
EINSUMS_FORCEINLINE bool bitwise_xor(B a, B b) {
    return a != b;
}
/// a && !b, as the vector bitwise_andnot is a & ~b.
template <std::same_as<bool> B>
EINSUMS_FORCEINLINE bool bitwise_andnot(B a, B b) {
    return a && !b;
}

// ---------------------------------------------------------------------------
// Scalar bits: bitcast between same-width types, and the logical shifts by an
// immediate count, as the vector forms.
// ---------------------------------------------------------------------------

template <detail::arithmetic To, detail::arithmetic From>
    requires(sizeof(To) == sizeof(From))
EINSUMS_FORCEINLINE To bitcast(From x) {
    return std::bit_cast<To>(x);
}
template <int S, std::integral T>
    requires(!std::same_as<T, bool>)
EINSUMS_FORCEINLINE T shift_left(T v) {
    return static_cast<T>(static_cast<std::make_unsigned_t<T>>(v) << S);
}
template <int S, std::integral T>
    requires(!std::same_as<T, bool>)
EINSUMS_FORCEINLINE T shift_right(T v) {
    return static_cast<T>(static_cast<std::make_unsigned_t<T>>(v) >> S);
}

// ---------------------------------------------------------------------------
// Scalar conversion.
// ---------------------------------------------------------------------------

/// x as To. Floating point to integer truncates toward zero and gives To's minimum for a NaN or an
/// out-of-range value, as x86 does, where a plain cast would be undefined. Integer to floating point
/// rounds to nearest, as do double to float and the vector conversions.
template <detail::arithmetic To, detail::arithmetic From>
EINSUMS_FORCEINLINE To convert(From x) {
    if constexpr (std::floating_point<From> && std::integral<To>) {
        // The range of To as From. Both limits are powers of two, so From holds them exactly.
        constexpr From low  = static_cast<From>(std::numeric_limits<To>::min());
        constexpr From high = static_cast<From>(std::numeric_limits<To>::max() / 2 + 1) * From{2};
        From const     t    = std::trunc(x);
        if (!(t >= low && t < high)) {
            return std::numeric_limits<To>::min();
        }
        return static_cast<To>(t);
    } else {
        return static_cast<To>(x);
    }
}

EINSUMS_SIMD_ISA_NAMESPACE_END()
EINSUMS_NAMESPACE_END(simd)
