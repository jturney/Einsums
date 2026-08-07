//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>

#include <iterator>
#include <tuple>

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Provides compile-time for loop over a sequence.
 *
 * @tparam T datatype of the sequence
 * @tparam S the sequence to work through
 * @tparam F the functor type to call
 * @param f the functor to call
 *
 * @versionadded{1.0.0}
 */
template <typename T, T... S, typename F>
constexpr void for_sequence(std::integer_sequence<T, S...>, F f) {
    (static_cast<void>(f(std::integral_constant<T, S>{})), ...);
}

/**
 * @brief Provides compile-time for loop semantics.
 *
 * Loops from 0 to n-1. This function expands as if it were a loop, but it does its
 * evaluation at compile time if possible. The functor is called with a
 * @c std::integral_constant on each iteration, whose value type is the same as the
 * type of @c n . Here is an example of how one might use this.
 *
 * @code
 * auto index_lists = std::make_tuple(Indices{index::i, index::j, index::k}, Indices{index::i, index::k, index::j});
 * Tensor<double, 3> A = create_random_tensor("A", 10, 10, 10), B = create_random_tensor("B", 10, 10, 10);
 * double C = 0;
 *
 * for_sequence<2>([&](auto n) { einsum(1.0, Indices{}, &C, 1.0,
 *     std::get<(size_t) n>(index_lists), A, Indices{index::i, index::j, index::k}, B); });
 * @endcode
 *
 * This code will go through and compute the einsum with several different layouts of
 * indices. Note that we need to cast the argument to the lambda to @c size_t so that
 * it can be consumed by @c std::get .
 *
 * @tparam n the number of iterations to perform
 * @tparam F the functor type to call
 * @param f the functor
 *
 * @versionadded{1.0.0}
 */
template <auto n, typename F>
constexpr void for_sequence(F f) {
    for_sequence(std::make_integer_sequence<decltype(n), n>{}, f);
}
/**
 * @brief Mimic Python's enumerate.
 *
 * @tparam T The iterable type.
 * @tparam Iter The iterator type for the iterable.
 * @param[in] iterable The iterable to iterate over.
 * @param[in] start The starting value for the number element. Defaults to zero. Does not affect the start point in the iterator.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Added the start parameter to match Python.
 * @endversion
 */
template <typename T, typename Iter = decltype(std::begin(std::declval<T>()))>
    requires requires(T iterable) {
        { std::end(iterable) } -> std::same_as<Iter>;
    }
constexpr auto enumerate(T &&iterable, ptrdiff_t start = 0) {
    struct Iterator {
        ptrdiff_t i;
        Iter      iter;

        constexpr auto operator!=(Iterator const &other) const -> bool { return iter != other.iter; }
        constexpr void operator++() {
            ++i;
            ++iter;
        }
        constexpr auto operator*() const { return std::tie(i, *iter); }
    };
    struct IterableWrapper {
        T              iterable;
        ptrdiff_t      start;
        constexpr auto begin() { return Iterator{start, std::begin(iterable)}; }
        constexpr auto end() { return Iterator{start, std::end(iterable)}; }
    };

    return IterableWrapper{std::forward<T>(iterable), start};
}

EINSUMS_NAMESPACE_END()
