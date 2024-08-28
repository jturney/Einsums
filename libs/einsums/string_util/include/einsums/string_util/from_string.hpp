//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/string_util/bad_lexical_cast.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace einsums::string_util {

namespace detail {

template <typename T, typename Enable = void>
struct from_string_impl {
    template <typename Char>
    static void call(std::basic_string<Char> const &value, T &target) {
        std::basic_istringstream<Char> stream(value);
        stream.exceptions(std::ios_base::failbit);
        stream >> target;
    }
};

template <typename T, typename U>
T check_out_of_range(U const &value) {
    U const max = (std::numeric_limits<T>::max)();
    if constexpr (std::is_unsigned_v<U>) {
        if (value > max) {
            throw std::out_of_range("from_string: out of range");
        }
    } else {
        U const min = (std::numeric_limits<T>::min)();
        if (value < min || value > max) {
            throw std::out_of_range("from_string: out of range");
        }
    }
    return static_cast<T>(value);
}

template <typename Char>
void check_only_whitespace(std::basic_string<Char> const &s, std::size_t pos) {
    auto i = s.begin();
    std::advance(i, pos);
    i = std::find_if(i, s.end(), [](int c) { return !std::isspace(c); });

    if (i != s.end()) {
        throw std::invalid_argument("from_string: found non-whitespace after token");
    }
}

template <typename T>
struct from_string_impl<T, std::enable_if_t<std::is_integral_v<T>>> {
    template <typename Char>
    static void call(std::basic_string<Char> const &value, int &target) {
        std::size_t pos = 0;
        target          = std::stoi(value, &pos);
        check_only_whitespace(value, pos);
    }

    template <typename Char>
    static void call(std::basic_string<Char> const &value, long &target) {
        std::size_t pos = 0;
        target          = std::stol(value, &pos);
        check_only_whitespace(value, pos);
    }

    template <typename Char>
    static void call(std::basic_string<Char> const &value, long long &target) {
        std::size_t pos = 0;
        target          = std::stoll(value, &pos);
        check_only_whitespace(value, pos);
    }

    template <typename Char>
    static void call(std::basic_string<Char> const &value, unsigned int &target) {
        // there is no std::stoui
        unsigned long target_long;
        call(value, target_long);
        target = check_out_of_range<T>(target_long);
    }

    template <typename Char>
    static void call(std::basic_string<Char> const &value, unsigned long &target) {
        std::size_t pos = 0;
        target          = std::stoul(value, &pos);
        check_only_whitespace(value, pos);
    }

    template <typename Char>
    static void call(std::basic_string<Char> const &value, unsigned long long &target) {
        std::size_t pos = 0;
        target          = std::stoull(value, &pos);
        check_only_whitespace(value, pos);
    }

    template <typename Char, typename U>
    static void call(std::basic_string<Char> const &value, U &target) {
        using promoted_t = decltype(+std::declval<U>());
        static_assert(!std::is_same_v<promoted_t, U>);

        promoted_t promoted;
        call(value, promoted);
        target = check_out_of_range<U>(promoted);
    }
};

template <typename T>
struct from_string_impl<T, std::enable_if_t<std::is_floating_point_v<T>>> {
    template <typename Char>
    static void call(std::basic_string<Char> const &value, float &target) {
        std::size_t pos = 0;
        target          = std::stof(value, &pos);
        check_only_whitespace(value, pos);
    }

    template <typename Char>
    static void call(std::basic_string<Char> const &value, double &target) {
        std::size_t pos = 0;
        target          = std::stod(value, &pos);
        check_only_whitespace(value, pos);
    }

    template <typename Char>
    static void call(std::basic_string<Char> const &value, long double &target) {
        std::size_t pos = 0;
        target          = std::stold(value, &pos);
        check_only_whitespace(value, pos);
    }
};

} // namespace detail

template <typename T, typename Char>
T from_string(std::basic_string<Char> const &v) {
    T target;
    try {
        detail::from_string_impl<T>::call(v, target);
    } catch (...) {
        return detail::throw_bad_lexical_cast<std::basic_string<Char>, T>();
    }
    return target;
}

template <typename T, typename U, typename Char>
T from_string(std::basic_string<Char> const &v, U &&default_value) {
    T target;
    try {
        detail::from_string_impl<T>::call(v, target);
        return target;
    } catch (...) {
        return std::forward<U>(default_value);
    }
}

template <typename T>
T from_string(std::string const &v) {
    T target;
    try {
        detail::from_string_impl<T>::call(v, target);
    } catch (...) {
        return detail::throw_bad_lexical_cast<std::string, T>();
    }
    return target;
}

template <typename T, typename U>
T from_string(std::string const &v, U &&default_value) {
    T target;
    try {
        detail::from_string_impl<T>::call(v, target);
        return target;
    } catch (...) {
        return std::forward<U>(default_value);
    }
}
} // namespace einsums::string_util