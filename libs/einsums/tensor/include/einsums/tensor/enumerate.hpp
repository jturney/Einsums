//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <iterator>

namespace einsums {

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
