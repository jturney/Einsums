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
//   scalar_t<V>, lanes_v<V>, is_vec_v<V>   element type, lane count, and whether V is a Vec
//   splat<V>(x)                           broadcast(x), or x
//   load<V>(p), store(p, v)               loadu and storeu, or *p
//   lookup(base, idx)                     gather(base, idx) with a lane per index, or base[idx]
//
// The scalar overloads match one lane of the vector operation bit for bit, so
// a scalar run on the CPU is an exact reference for the vector run: min is
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
template <typename T>
struct is_vec<Vec<T>> : std::true_type {};

template <typename V>
struct lanes_of : std::integral_constant<int, 1> {};
template <typename T>
struct lanes_of<Vec<T>> : std::integral_constant<int, Vec<T>::lanes> {};

template <typename V>
struct value_of {
    using type = V;
};
template <typename T>
struct value_of<Vec<T>> {
    using type = typename Vec<T>::value_type;
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
    if constexpr (is_vec_v<V>) {
        return broadcast(x);
    } else {
        return x;
    }
}

/// lanes_v<V> consecutive elements from p, which need no alignment.
template <typename V>
EINSUMS_FORCEINLINE V load(scalar_t<V> const *p) {
    if constexpr (is_vec_v<V>) {
        return loadu(p);
    } else {
        return *p;
    }
}

/// Write v's lanes to consecutive elements from p, which needs no alignment.
template <typename T>
EINSUMS_FORCEINLINE void store(T *p, Vec<T> v) {
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
