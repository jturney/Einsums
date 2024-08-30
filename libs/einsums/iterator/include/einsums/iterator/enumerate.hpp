//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <iterator>

namespace einsums {

/**
 * @brief Provides compile-time for loop over a sequence.
 *
 * @tparam T datatype of the sequence
 * @tparam S the sequence to work through
 * @tparam F the functor type to call
 * @param f the functor to call
 */
template <typename T, T... S, typename F>
constexpr void for_sequence(std::integer_sequence<T, S...>, F f) {
    (static_cast<void>(f(std::integral_constant<T, S>{})), ...);
}

/**
 * @brief Provides compile-time for loop semantics.
 *
 * Loops from 0 to n-1.
 *
 * @tparam n the number of iterations to perform
 * @tparam F the functor type to call
 * @param f the functor
 */
template <auto n, typename F>
constexpr void for_sequence(F f) {
    for_sequence(std::make_integer_sequence<decltype(n), n>{}, f);
}
/// Mimic Python's enumerate.
template <typename T, typename Iter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))> // The type of the end isn't needed but we must ensure
                                                            // it is valid.
constexpr auto enumerate(T &&iterable) {
    struct Iterator {
        std::size_t i;
        Iter        iter;

        auto operator!=(const Iterator &other) const -> bool { return iter != other.iter; }
        void operator++() {
            ++i;
            ++iter;
        }
        auto operator*() const { return std::tie(i, *iter); }
    };
    struct IterableWrapper {
        T    iterable;
        auto begin() { return Iterator{0, std::begin(iterable)}; }
        auto end() { return Iterator{0, std::end(iterable)}; }
    };

    return IterableWrapper{std::forward<T>(iterable)};
}

} // namespace einsums
