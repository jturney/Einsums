//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/NamedRequirements.hpp>

#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <utility>

namespace einsums {

namespace detail {
template <typename T, typename Haystack, size_t... __I>
    requires requires {
        { std::tuple_size_v<Haystack> };
        { std::get<0>(std::declval<Haystack>()) };
    }
constexpr bool is_in(T &&needle, Haystack const &haystack, std::index_sequence<__I...> const &) {
    return ((needle == std::get<__I>(haystack)) || ... || false);
}

template <typename T>
concept IsTupleOrArray = requires {
    { std::tuple_size_v<T> };
    { std::get<0>(std::declval<T>()) };
};
} // namespace detail

/**
 * @brief Check to see if an element is contained in a container or tuple.
 *
 * @versionadded{1.1.0}
 */
template <typename T, detail::IsTupleOrArray Haystack>
constexpr bool is_in(T &&needle, Haystack const &haystack) {
    return detail::is_in(std::forward<T>(needle), haystack, std::make_index_sequence<std::tuple_size_v<Haystack>>());
}

/**
 * @brief Check to see if an element is contained in a container.
 *
 * @versionadded{1.1.0}
 */
template <typename T, Container Haystack>
    requires(!detail::IsTupleOrArray<Haystack>)
constexpr bool is_in(T &&needle, Haystack const &haystack) {
    for (auto test : haystack) {
        if (needle == test) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Check to see if an element is contained in an initializer list.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
constexpr bool is_in(T &&needle, std::initializer_list<std::decay_t<T>> haystack) {
    for (auto test : haystack) {
        if (needle == test) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Check to see if an element is not contained in a container or tuple.
 *
 * @versionadded{1.1.0}
 */
template <typename T, detail::IsTupleOrArray Haystack>
constexpr bool not_in(T &&needle, Haystack const &haystack) {
    return !detail::is_in(std::forward<T>(needle), std::forward<Haystack>(haystack),
                          std::make_index_sequence<std::tuple_size_v<Haystack>>());
}

/**
 * @brief Check to see if an element is not contained in a container.
 *
 * @versionadded{1.1.0}
 */
template <typename T, Container Haystack>
    requires(!detail::IsTupleOrArray<Haystack>)
constexpr bool not_in(T &&needle, Haystack const &haystack) {
    for (auto test : haystack) {
        if (needle == test) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Check to see if an element is not contained in an initializer list.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
constexpr bool not_in(T &&needle, std::initializer_list<std::decay_t<T>> haystack) {
    for (auto test : haystack) {
        if (needle == test) {
            return false;
        }
    }
    return true;
}

} // namespace einsums