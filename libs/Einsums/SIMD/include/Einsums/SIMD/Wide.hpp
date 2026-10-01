//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/SIMD/Convert.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Partial.hpp>
#include <Einsums/SIMD/Reduce.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

// ===========================================================================
// Wide vectors: Vec<T, N> and Mask<T, N> with N a multiple K of the native
// lane count, held as K native parts in lane order (see Vec.hpp).
//
// Every operation here is the native operation applied to each part, so a wide
// result is bit for bit the native results side by side; a reduction folds the
// parts together first. The use is mixed precision at equal width: the FP64
// tier of a kernel is Vec<double, lanes<float>>, which holds as many lanes as
// the FP32 tier's Vec<float>.
//
// Moving between such tiers is the same-lane conversion, now for any two
// element types with the same lane count N:
//
//   convert<double>(Vec<float, N>)      -> Vec<double, N>   widen, exact
//   convert<float>(Vec<double, N>)      -> Vec<float, N>    round to nearest even
//   convert<int32_t>(Vec<double, N>)    -> Vec<int32_t, N>  truncate
//   convert<double>(Vec<int32_t, N>)    -> Vec<double, N>
//   mask_cast<double>(Mask<float, N>)   -> Mask<double, N>  lane i stays lane i
//   mask_cast<float>(Mask<double, N>)   -> Mask<float, N>
//
// The conversions are the paired ones of Convert.hpp, one per register; the
// mask casts are one or two instructions per register on every ISA.
// ===========================================================================

EINSUMS_NAMESPACE_BEGIN(simd)
EINSUMS_SIMD_ISA_NAMESPACE_BEGIN()

namespace detail {
/// N lanes of T need more than one register.
template <typename T, int N>
concept wide_lanes = N != VecTraits<T>::lanes;

/// Part k of a vector or mask, which is the whole of a native one.
template <typename T, int N>
EINSUMS_FORCEINLINE Vec<T> part_of(Vec<T, N> const &v, int k) {
    if constexpr (Vec<T, N>::native) {
        return v;
    } else {
        return v.part[k];
    }
}
template <typename T, int N>
EINSUMS_FORCEINLINE void set_part(Vec<T, N> &v, int k, Vec<T> p) {
    if constexpr (Vec<T, N>::native) {
        v = p;
    } else {
        v.part[k] = p;
    }
}
template <typename T, int N>
EINSUMS_FORCEINLINE Mask<T> part_of(Mask<T, N> const &m, int k) {
    if constexpr (Mask<T, N>::native) {
        return m;
    } else {
        return m.part[k];
    }
}
template <typename T, int N>
EINSUMS_FORCEINLINE void set_part(Mask<T, N> &m, int k, Mask<T> p) {
    if constexpr (Mask<T, N>::native) {
        m = p;
    } else {
        m.part[k] = p;
    }
}
} // namespace detail

// ---------------------------------------------------------------------------
// Elementwise operations, comparisons and select.
// ---------------------------------------------------------------------------

#define EINSUMS_SIMD_WIDE_UNARY(fn)                                                                                                        \
    template <typename T, int N>                                                                                                           \
        requires detail::wide_lanes<T, N>                                                                                                  \
    EINSUMS_FORCEINLINE Vec<T, N> fn(Vec<T, N> a) {                                                                                        \
        Vec<T, N> r;                                                                                                                       \
        for (int k = 0; k < Vec<T, N>::parts; ++k) {                                                                                       \
            r.part[k] = fn(a.part[k]);                                                                                                     \
        }                                                                                                                                  \
        return r;                                                                                                                          \
    }
#define EINSUMS_SIMD_WIDE_BINARY(fn)                                                                                                       \
    template <typename T, int N>                                                                                                           \
        requires detail::wide_lanes<T, N>                                                                                                  \
    EINSUMS_FORCEINLINE Vec<T, N> fn(Vec<T, N> a, Vec<T, N> b) {                                                                           \
        Vec<T, N> r;                                                                                                                       \
        for (int k = 0; k < Vec<T, N>::parts; ++k) {                                                                                       \
            r.part[k] = fn(a.part[k], b.part[k]);                                                                                          \
        }                                                                                                                                  \
        return r;                                                                                                                          \
    }
#define EINSUMS_SIMD_WIDE_TERNARY(fn)                                                                                                      \
    template <typename T, int N>                                                                                                           \
        requires detail::wide_lanes<T, N>                                                                                                  \
    EINSUMS_FORCEINLINE Vec<T, N> fn(Vec<T, N> a, Vec<T, N> b, Vec<T, N> c) {                                                              \
        Vec<T, N> r;                                                                                                                       \
        for (int k = 0; k < Vec<T, N>::parts; ++k) {                                                                                       \
            r.part[k] = fn(a.part[k], b.part[k], c.part[k]);                                                                               \
        }                                                                                                                                  \
        return r;                                                                                                                          \
    }
#define EINSUMS_SIMD_WIDE_COMPARE(fn)                                                                                                      \
    template <typename T, int N>                                                                                                           \
        requires detail::wide_lanes<T, N>                                                                                                  \
    EINSUMS_FORCEINLINE Mask<T, N> fn(Vec<T, N> a, Vec<T, N> b) {                                                                          \
        Mask<T, N> r;                                                                                                                      \
        for (int k = 0; k < Vec<T, N>::parts; ++k) {                                                                                       \
            r.part[k] = fn(a.part[k], b.part[k]);                                                                                          \
        }                                                                                                                                  \
        return r;                                                                                                                          \
    }

EINSUMS_SIMD_WIDE_UNARY(neg)
EINSUMS_SIMD_WIDE_UNARY(sqrt)
EINSUMS_SIMD_WIDE_UNARY(abs)
EINSUMS_SIMD_WIDE_UNARY(floor)
EINSUMS_SIMD_WIDE_UNARY(ceil)
EINSUMS_SIMD_WIDE_UNARY(trunc)
EINSUMS_SIMD_WIDE_UNARY(round)
EINSUMS_SIMD_WIDE_UNARY(round_even)
EINSUMS_SIMD_WIDE_BINARY(add)
EINSUMS_SIMD_WIDE_BINARY(sub)
EINSUMS_SIMD_WIDE_BINARY(mul)
EINSUMS_SIMD_WIDE_BINARY(div)
EINSUMS_SIMD_WIDE_BINARY(min)
EINSUMS_SIMD_WIDE_BINARY(max)
EINSUMS_SIMD_WIDE_BINARY(bitwise_and)
EINSUMS_SIMD_WIDE_BINARY(bitwise_or)
EINSUMS_SIMD_WIDE_BINARY(bitwise_xor)
EINSUMS_SIMD_WIDE_BINARY(bitwise_andnot)
EINSUMS_SIMD_WIDE_TERNARY(fmadd)
EINSUMS_SIMD_WIDE_TERNARY(fmsub)
EINSUMS_SIMD_WIDE_TERNARY(fnmadd)
EINSUMS_SIMD_WIDE_TERNARY(fnmsub)
EINSUMS_SIMD_WIDE_COMPARE(cmp_eq)
EINSUMS_SIMD_WIDE_COMPARE(cmp_ne)
EINSUMS_SIMD_WIDE_COMPARE(cmp_lt)
EINSUMS_SIMD_WIDE_COMPARE(cmp_le)
EINSUMS_SIMD_WIDE_COMPARE(cmp_gt)
EINSUMS_SIMD_WIDE_COMPARE(cmp_ge)
#undef EINSUMS_SIMD_WIDE_UNARY
#undef EINSUMS_SIMD_WIDE_BINARY
#undef EINSUMS_SIMD_WIDE_TERNARY
#undef EINSUMS_SIMD_WIDE_COMPARE

template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Vec<T, N> select(Mask<T, N> m, Vec<T, N> a, Vec<T, N> b) {
    Vec<T, N> r;
    for (int k = 0; k < Vec<T, N>::parts; ++k) {
        r.part[k] = select(m.part[k], a.part[k], b.part[k]);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Mask logic.
// ---------------------------------------------------------------------------

#define EINSUMS_SIMD_WIDE_MASK_BINARY(fn)                                                                                                  \
    template <typename T, int N>                                                                                                           \
        requires detail::wide_lanes<T, N>                                                                                                  \
    EINSUMS_FORCEINLINE Mask<T, N> fn(Mask<T, N> a, Mask<T, N> b) {                                                                        \
        Mask<T, N> r;                                                                                                                      \
        for (int k = 0; k < Mask<T, N>::parts; ++k) {                                                                                      \
            r.part[k] = fn(a.part[k], b.part[k]);                                                                                          \
        }                                                                                                                                  \
        return r;                                                                                                                          \
    }
EINSUMS_SIMD_WIDE_MASK_BINARY(bitwise_and)
EINSUMS_SIMD_WIDE_MASK_BINARY(bitwise_or)
EINSUMS_SIMD_WIDE_MASK_BINARY(bitwise_xor)
EINSUMS_SIMD_WIDE_MASK_BINARY(bitwise_andnot)
#undef EINSUMS_SIMD_WIDE_MASK_BINARY

template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Mask<T, N> operator&(Mask<T, N> a, Mask<T, N> b) {
    return bitwise_and(a, b);
}
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Mask<T, N> operator|(Mask<T, N> a, Mask<T, N> b) {
    return bitwise_or(a, b);
}
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Mask<T, N> operator^(Mask<T, N> a, Mask<T, N> b) {
    return bitwise_xor(a, b);
}
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Mask<T, N> operator!(Mask<T, N> a) {
    Mask<T, N> r;
    for (int k = 0; k < Mask<T, N>::parts; ++k) {
        r.part[k] = !a.part[k];
    }
    return r;
}
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Mask<T, N> &operator&=(Mask<T, N> &a, Mask<T, N> b) {
    return a = a & b;
}
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Mask<T, N> &operator|=(Mask<T, N> &a, Mask<T, N> b) {
    return a = a | b;
}
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Mask<T, N> &operator^=(Mask<T, N> &a, Mask<T, N> b) {
    return a = a ^ b;
}

/// The lanes of m as an integer, lane i in bit i.
template <typename T, int N>
    requires(detail::wide_lanes<T, N> && N <= 64)
EINSUMS_FORCEINLINE uint64_t to_bits(Mask<T, N> m) {
    constexpr int L    = VecTraits<T>::lanes;
    uint64_t      bits = 0;
    for (int k = 0; k < Mask<T, N>::parts; ++k) {
        bits |= to_bits(m.part[k]) << (k * L);
    }
    return bits;
}
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE bool any(Mask<T, N> m) {
    bool r = false;
    for (int k = 0; k < Mask<T, N>::parts; ++k) {
        r = r || any(m.part[k]);
    }
    return r;
}
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE bool all(Mask<T, N> m) {
    bool r = true;
    for (int k = 0; k < Mask<T, N>::parts; ++k) {
        r = r && all(m.part[k]);
    }
    return r;
}
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE bool none(Mask<T, N> m) {
    return !any(m);
}
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE int count(Mask<T, N> m) {
    int r = 0;
    for (int k = 0; k < Mask<T, N>::parts; ++k) {
        r += count(m.part[k]);
    }
    return r;
}
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Vec<T, N> to_vec(Mask<T, N> m) {
    Vec<T, N> r;
    for (int k = 0; k < Mask<T, N>::parts; ++k) {
        r.part[k] = to_vec(m.part[k]);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Operators.
// ---------------------------------------------------------------------------

#if !defined(EINSUMS_SIMD_NO_OPERATORS)
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Vec<T, N> operator-(Vec<T, N> a) {
    return neg(a);
}
#    define EINSUMS_SIMD_WIDE_OPERATOR(op, fn)                                                                                             \
        template <typename T, int N>                                                                                                       \
            requires detail::wide_lanes<T, N>                                                                                              \
        EINSUMS_FORCEINLINE Vec<T, N> operator op(Vec<T, N> a, Vec<T, N> b) {                                                              \
            return fn(a, b);                                                                                                               \
        }                                                                                                                                  \
        template <typename T, int N, detail::scalar_operand_for<T> S>                                                                      \
            requires detail::wide_lanes<T, N>                                                                                              \
        EINSUMS_FORCEINLINE Vec<T, N> operator op(Vec<T, N> a, S b) {                                                                      \
            Vec<T, N> r;                                                                                                                   \
            for (int k = 0; k < Vec<T, N>::parts; ++k) {                                                                                   \
                r.part[k] = fn(a.part[k], broadcast(static_cast<T>(b)));                                                                   \
            }                                                                                                                              \
            return r;                                                                                                                      \
        }                                                                                                                                  \
        template <typename T, int N, detail::scalar_operand_for<T> S>                                                                      \
            requires detail::wide_lanes<T, N>                                                                                              \
        EINSUMS_FORCEINLINE Vec<T, N> operator op(S a, Vec<T, N> b) {                                                                      \
            Vec<T, N> r;                                                                                                                   \
            for (int k = 0; k < Vec<T, N>::parts; ++k) {                                                                                   \
                r.part[k] = fn(broadcast(static_cast<T>(a)), b.part[k]);                                                                   \
            }                                                                                                                              \
            return r;                                                                                                                      \
        }                                                                                                                                  \
        template <typename T, int N>                                                                                                       \
            requires detail::wide_lanes<T, N>                                                                                              \
        EINSUMS_FORCEINLINE Vec<T, N> &operator op## = (Vec<T, N> & a, Vec<T, N> b) {                                                      \
            return a = fn(a, b);                                                                                                           \
        }                                                                                                                                  \
        template <typename T, int N, detail::scalar_operand_for<T> S>                                                                      \
            requires detail::wide_lanes<T, N>                                                                                              \
        EINSUMS_FORCEINLINE Vec<T, N> &operator op## = (Vec<T, N> & a, S b) {                                                              \
            return a = a op b;                                                                                                             \
        }
EINSUMS_SIMD_WIDE_OPERATOR(+, add)
EINSUMS_SIMD_WIDE_OPERATOR(-, sub)
EINSUMS_SIMD_WIDE_OPERATOR(*, mul)
EINSUMS_SIMD_WIDE_OPERATOR(/, div)
#    undef EINSUMS_SIMD_WIDE_OPERATOR
#endif

// ---------------------------------------------------------------------------
// Memory and reductions.
// ---------------------------------------------------------------------------

/// Write v's lanes to consecutive elements from p, which needs no alignment.
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE void storeu(T *p, Vec<T, N> v) {
    constexpr int L = VecTraits<T>::lanes;
    for (int k = 0; k < Vec<T, N>::parts; ++k) {
        storeu(p + k * L, v.part[k]);
    }
}

/// p[i] in each lane m sets, zero elsewhere; an inactive lane's memory is never read.
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Vec<T, N> loadu(T const *p, Mask<T, N> m) {
    constexpr int L = VecTraits<T>::lanes;
    Vec<T, N>     r;
    for (int k = 0; k < Vec<T, N>::parts; ++k) {
        r.part[k] = loadu(p + k * L, m.part[k]);
    }
    return r;
}

/// Write p[i] for each lane m sets, and nothing else.
template <typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE void storeu(T *p, Vec<T, N> v, Mask<T, N> m) {
    constexpr int L = VecTraits<T>::lanes;
    for (int k = 0; k < Vec<T, N>::parts; ++k) {
        storeu(p + k * L, v.part[k], m.part[k]);
    }
}

#define EINSUMS_SIMD_WIDE_REDUCE(fn, combine)                                                                                              \
    template <typename T, int N>                                                                                                           \
        requires detail::wide_lanes<T, N>                                                                                                  \
    EINSUMS_FORCEINLINE T fn(Vec<T, N> v) {                                                                                                \
        Vec<T> folded = v.part[0];                                                                                                         \
        for (int k = 1; k < Vec<T, N>::parts; ++k) {                                                                                       \
            folded = combine(folded, v.part[k]);                                                                                           \
        }                                                                                                                                  \
        return fn(folded);                                                                                                                 \
    }
EINSUMS_SIMD_WIDE_REDUCE(reduce_add, add)
EINSUMS_SIMD_WIDE_REDUCE(reduce_min, min)
EINSUMS_SIMD_WIDE_REDUCE(reduce_max, max)
#undef EINSUMS_SIMD_WIDE_REDUCE

// ---------------------------------------------------------------------------
// Bits: bitcast between same-width element types, and the immediate shifts.
// ---------------------------------------------------------------------------

template <typename To, typename From, int N>
    requires(detail::wide_lanes<From, N> && sizeof(To) == sizeof(From))
EINSUMS_FORCEINLINE Vec<To, N> bitcast(Vec<From, N> v) {
    Vec<To, N> r;
    for (int k = 0; k < Vec<From, N>::parts; ++k) {
        r.part[k] = bitcast<To>(v.part[k]);
    }
    return r;
}
template <int S, typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Vec<T, N> shift_left(Vec<T, N> v) {
    Vec<T, N> r;
    for (int k = 0; k < Vec<T, N>::parts; ++k) {
        r.part[k] = shift_left<S>(v.part[k]);
    }
    return r;
}
template <int S, typename T, int N>
    requires detail::wide_lanes<T, N>
EINSUMS_FORCEINLINE Vec<T, N> shift_right(Vec<T, N> v) {
    Vec<T, N> r;
    for (int k = 0; k < Vec<T, N>::parts; ++k) {
        r.part[k] = shift_right<S>(v.part[k]);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Moving between element types with the same lane count.
// ---------------------------------------------------------------------------

/// The same lanes as another element type, for any two types with the same lane count N; see the top
/// of this header. The form for two types of one native lane count is in Convert.hpp.
template <typename To, typename From, int N>
    requires(N != VecTraits<To>::lanes || N != VecTraits<From>::lanes)
EINSUMS_FORCEINLINE Vec<To, N> convert(Vec<From, N> v) {
    constexpr int LF = VecTraits<From>::lanes;
    constexpr int LT = VecTraits<To>::lanes;
    Vec<To, N>    r;
    if constexpr (LF == LT) {
        for (int k = 0; k < N / LF; ++k) {
            detail::set_part(r, k, convert<To>(detail::part_of(v, k)));
        }
    } else if constexpr (LF == 2 * LT) {
        // Each source register widens into two.
        for (int k = 0; k < N / LF; ++k) {
            Vec<From> const p = detail::part_of(v, k);
            detail::set_part(r, 2 * k, convert_low<To>(p));
            detail::set_part(r, 2 * k + 1, convert_high<To>(p));
        }
    } else if constexpr (LT == 2 * LF) {
        // Each pair of source registers narrows into one.
        for (int k = 0; k < N / LT; ++k) {
            detail::set_part(r, k, convert<To>(detail::part_of(v, 2 * k), detail::part_of(v, 2 * k + 1)));
        }
    } else {
        static_assert(LF == LT, "convert moves between element types of equal width, or of twice or half the width");
    }
    return r;
}

namespace detail {
// One register of 32-bit lane flags as two registers of 64-bit lane flags, and back. Lane i stays lane
// i: the low register takes lanes 0 .. L/2 - 1.
template <typename Wide, typename Narrow>
EINSUMS_FORCEINLINE void mask_widen(Mask<Narrow> m, Mask<Wide> &lo, Mask<Wide> &hi) {
#if defined(EINSUMS_SIMD_MASK_IS_K)
    using K = typename Mask<Wide>::reg_type;
    lo      = Mask<Wide>(static_cast<K>(m.reg & 0xFFu));
    hi      = Mask<Wide>(static_cast<K>(m.reg >> 8));
#elif defined(__AVX__)
    __m256 const f  = std::bit_cast<__m256>(m.reg);
    __m256 const ul = _mm256_unpacklo_ps(f, f); // lanes 0 1 | 4 5, each doubled
    __m256 const uh = _mm256_unpackhi_ps(f, f); // lanes 2 3 | 6 7, each doubled
    lo              = Mask<Wide>(std::bit_cast<typename Mask<Wide>::reg_type>(_mm256_permute2f128_ps(ul, uh, 0x20)));
    hi              = Mask<Wide>(std::bit_cast<typename Mask<Wide>::reg_type>(_mm256_permute2f128_ps(ul, uh, 0x31)));
#elif defined(__aarch64__) || defined(_M_ARM64)
    int32x4_t const s = vreinterpretq_s32_u32(m.reg);
    lo                = Mask<Wide>(vreinterpretq_u64_s64(vmovl_s32(vget_low_s32(s))));
    hi                = Mask<Wide>(vreinterpretq_u64_s64(vmovl_high_s32(s)));
#elif defined(EINSUMS_SIMD_MASK_IS_VECTOR)
    __m128 const f = std::bit_cast<__m128>(m.reg);
    lo             = Mask<Wide>(std::bit_cast<typename Mask<Wide>::reg_type>(_mm_unpacklo_ps(f, f)));
    hi             = Mask<Wide>(std::bit_cast<typename Mask<Wide>::reg_type>(_mm_unpackhi_ps(f, f)));
#else
    static_assert(sizeof(Wide) == 0, "the scalar build has equal lane counts and never widens a mask");
#endif
}

template <typename Narrow, typename Wide>
EINSUMS_FORCEINLINE Mask<Narrow> mask_narrow(Mask<Wide> lo, Mask<Wide> hi) {
#if defined(EINSUMS_SIMD_MASK_IS_K)
    using K = typename Mask<Narrow>::reg_type;
    return Mask<Narrow>(static_cast<K>(static_cast<unsigned>(lo.reg) | (static_cast<unsigned>(hi.reg) << 8)));
#elif defined(__AVX__)
    // Every other 32-bit word of each 64-bit lane flag, in order.
    __m256 const l = std::bit_cast<__m256>(lo.reg), h = std::bit_cast<__m256>(hi.reg);
    __m128 const a = _mm_shuffle_ps(_mm256_castps256_ps128(l), _mm256_extractf128_ps(l, 1), _MM_SHUFFLE(2, 0, 2, 0));
    __m128 const b = _mm_shuffle_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1), _MM_SHUFFLE(2, 0, 2, 0));
    return Mask<Narrow>(std::bit_cast<typename Mask<Narrow>::reg_type>(_mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1)));
#elif defined(__aarch64__) || defined(_M_ARM64)
    return Mask<Narrow>(vcombine_u32(vmovn_u64(lo.reg), vmovn_u64(hi.reg)));
#elif defined(EINSUMS_SIMD_MASK_IS_VECTOR)
    __m128 const l = std::bit_cast<__m128>(lo.reg), h = std::bit_cast<__m128>(hi.reg);
    return Mask<Narrow>(std::bit_cast<typename Mask<Narrow>::reg_type>(_mm_shuffle_ps(l, h, _MM_SHUFFLE(2, 0, 2, 0))));
#else
    static_assert(sizeof(Wide) == 0, "the scalar build has equal lane counts and never narrows a mask");
#endif
}
} // namespace detail

/// The same lanes as a mask of another element type, for any two types with the same lane count N:
/// between widths it widens or narrows, lane i staying lane i. The form for one native lane count is
/// in Operations.hpp.
template <typename U, typename T, int N>
    requires(N != VecTraits<U>::lanes || N != VecTraits<T>::lanes || sizeof(U) != sizeof(T))
EINSUMS_FORCEINLINE Mask<U, N> mask_cast(Mask<T, N> m) {
    constexpr int LT = VecTraits<T>::lanes;
    constexpr int LU = VecTraits<U>::lanes;
    Mask<U, N>    r;
    if constexpr (LT == LU) {
        for (int k = 0; k < N / LT; ++k) {
            if constexpr (sizeof(U) == sizeof(T)) {
                detail::set_part(r, k, mask_cast<U>(detail::part_of(m, k)));
            } else {
                // Only the scalar build has equal lane counts across widths; there a mask is a bool.
                detail::set_part(r, k, Mask<U>(detail::part_of(m, k).reg));
            }
        }
    } else if constexpr (LT == 2 * LU) {
        for (int k = 0; k < N / LT; ++k) {
            Mask<U> lo, hi;
            detail::mask_widen<U, T>(detail::part_of(m, k), lo, hi);
            detail::set_part(r, 2 * k, lo);
            detail::set_part(r, 2 * k + 1, hi);
        }
    } else if constexpr (LU == 2 * LT) {
        for (int k = 0; k < N / LU; ++k) {
            detail::set_part(r, k, detail::mask_narrow<U, T>(detail::part_of(m, 2 * k), detail::part_of(m, 2 * k + 1)));
        }
    } else {
        static_assert(LT == LU, "mask_cast moves between element types of equal width, or of twice or half the width");
    }
    return r;
}

EINSUMS_SIMD_ISA_NAMESPACE_END()
EINSUMS_NAMESPACE_END(simd)
