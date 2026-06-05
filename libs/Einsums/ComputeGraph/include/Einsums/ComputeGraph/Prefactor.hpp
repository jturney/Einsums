//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file Prefactor.hpp
 * @brief Type-erased scalar for einsum / scale prefactors.
 *
 * The graph's OpData variant carries one EinsumDescriptor type; templating it
 * on the scalar would explode the variant or force every Node/Graph to be
 * templated. Instead, the prefactor itself is type-erased into a variant over
 * the four bound dtypes. Optimization passes that read the prefactor use
 * the helpers below; the executor lambda — which already knows the concrete
 * tensor type — calls @ref as<T> to extract a typed scalar for BLAS dispatch.
 */

#include <fmt/format.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>

namespace einsums::compute_graph {

/// Type-erased prefactor scalar. Default-constructs to ``double{0}``.
using PrefactorScalar = std::variant<float, double, std::complex<float>, std::complex<double>>;

/// True iff the contained value compares equal to zero.
inline bool is_zero(PrefactorScalar const &v) {
    return std::visit([](auto x) { return x == decltype(x){0}; }, v);
}

/// True iff the contained value compares equal to one.
inline bool is_one(PrefactorScalar const &v) {
    return std::visit([](auto x) { return x == decltype(x){1}; }, v);
}

/// Convert a PrefactorScalar to a concrete typed scalar.
///
/// Real → real and real → complex are exact. Complex → real throws when the
/// imaginary part is non-zero (loud failure for type confusion); use
/// @ref as_real to take ``.real()`` lossily on purpose.
///
/// The body branches on @c T first so that ``typename T::value_type`` is
/// only ever named inside a branch where @c T is known to be a complex
/// type. Substitution-time well-formedness checking in some clang versions
/// otherwise rejects the discarded branch.
template <typename T>
T as(PrefactorScalar const &v) {
    return std::visit(
        [](auto x) -> T {
            using U = decltype(x);
            if constexpr (std::is_arithmetic_v<T>) {
                // Target is a real scalar.
                if constexpr (std::is_arithmetic_v<U>) {
                    return static_cast<T>(x);
                } else {
                    if (x.imag() != typename U::value_type{0}) {
                        throw std::runtime_error("PrefactorScalar: lossy complex→real conversion (imag != 0); use as_real() to opt in");
                    }
                    return static_cast<T>(x.real());
                }
            } else {
                // Target is a complex scalar.
                using R = typename T::value_type;
                if constexpr (std::is_arithmetic_v<U>) {
                    return T{static_cast<R>(x), R{0}};
                } else {
                    return T{static_cast<R>(x.real()), static_cast<R>(x.imag())};
                }
            }
        },
        v);
}

/// Lossy projection to a real scalar. Discards any imaginary component.
template <typename T>
T as_real(PrefactorScalar const &v) {
    static_assert(std::is_arithmetic_v<T>, "as_real<T>: T must be a real arithmetic type");
    return std::visit(
        [](auto x) -> T {
            using U = decltype(x);
            if constexpr (std::is_arithmetic_v<U>) {
                return static_cast<T>(x);
            } else {
                return static_cast<T>(x.real());
            }
        },
        v);
}

/// Hash a PrefactorScalar by its raw bits (per alternative). Stable across
/// runs and discriminates between alternatives via the variant index.
inline std::size_t hash(PrefactorScalar const &v) {
    std::size_t h = v.index();
    std::visit(
        [&h](auto x) {
            using U = decltype(x);
            if constexpr (std::is_arithmetic_v<U>) {
                if constexpr (sizeof(U) == sizeof(std::uint32_t)) {
                    std::uint32_t bits;
                    std::memcpy(&bits, &x, sizeof(bits));
                    h ^= std::hash<std::uint32_t>{}(bits) + 0x9e3779b9 + (h << 6) + (h >> 2);
                } else {
                    std::uint64_t bits;
                    std::memcpy(&bits, &x, sizeof(bits));
                    h ^= std::hash<std::uint64_t>{}(bits) + 0x9e3779b9 + (h << 6) + (h >> 2);
                }
            } else {
                auto re = x.real();
                auto im = x.imag();
                if constexpr (sizeof(re) == sizeof(std::uint32_t)) {
                    std::uint32_t rb, ib;
                    std::memcpy(&rb, &re, sizeof(rb));
                    std::memcpy(&ib, &im, sizeof(ib));
                    h ^= std::hash<std::uint32_t>{}(rb) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= std::hash<std::uint32_t>{}(ib) + 0x9e3779b9 + (h << 6) + (h >> 2);
                } else {
                    std::uint64_t rb, ib;
                    std::memcpy(&rb, &re, sizeof(rb));
                    std::memcpy(&ib, &im, sizeof(ib));
                    h ^= std::hash<std::uint64_t>{}(rb) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= std::hash<std::uint64_t>{}(ib) + 0x9e3779b9 + (h << 6) + (h >> 2);
                }
            }
        },
        v);
    return h;
}

/// Format a PrefactorScalar for logs / JSON. Real values print as-is; complex
/// as ``(re,im)``. Used by profile annotations and graph JSON serialization.
inline std::string to_string(PrefactorScalar const &v) {
    return std::visit(
        [](auto x) -> std::string {
            using U = decltype(x);
            if constexpr (std::is_arithmetic_v<U>) {
                return fmt::format("{}", x);
            } else {
                return fmt::format("({},{})", x.real(), x.imag());
            }
        },
        v);
}

} // namespace einsums::compute_graph
