//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/StringUtil/BadLexicalCast.hpp>

#include <algorithm>

namespace einsums {

/**
 * @struct from_string_impl
 *
 * @brief Converts a string to a value.
 *
 * @versionadded{1.0.0}
 */
template <typename T, typename Enable = void>
struct FromStringImpl {

    /**
     * @brief Performs the string conversion.
     *
     * @param value The string to read.
     * @param target The output variable.
     *
     * @versionadded{1.0.0}
     */
    template <typename Char>
    static void call(std::basic_string<Char> const &value, T &target) {
        std::basic_istringstream<Char> stream(value);
        stream.exceptions(std::ios_base::failbit);
        stream >> target;
    }
};

/**
 * @brief Check if the value is within the range of the requested type.
 *
 * @tparam T The type to check. This determines the maximum and minimum values.
 * @tparam U The type of the input.
 * @param value The value to check.
 * @return Casts the value into the new type.
 *
 * @throws std::out_of_range Throws this if the value is outside of the representable range of the check type.
 *
 * @versionadded{1.0.0}
 */
template <typename T, typename U>
T check_out_of_range(U const &value) {
    U const max = (std::numeric_limits<T>::max)();
    if constexpr (std::is_unsigned_v<U>) {
        if (value > max) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "from_string: out of range");
        }
    } else {
        U const min = (std::numeric_limits<T>::min)();
        if (value < min || value > max) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "from_string: out of range");
        }
    }
    return static_cast<T>(value);
}

/**
 * @brief Checks to see if all the characters in the string past a certain point are all whitespace.
 *
 * @tparam Char The kind of character.
 * @param s The string to check.
 * @param pos The position to start checking from.
 *
 * @throws std::invalid_argument Throws if there are non-whitespace characters after the requested position.
 *
 * @versionadded{1.0.0}
 */
template <typename Char>
void check_only_whitespace(std::basic_string<Char> const &s, std::size_t pos) {
    auto i = s.begin();
    std::advance(i, pos);
    i = std::find_if(i, s.end(), [](int c) { return !std::isspace(c); });

    if (i != s.end()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "from_string: found non-whitespace after token");
    }
}

#ifndef DOXYGEN
template <typename T>
struct FromStringImpl<T, std::enable_if_t<std::is_integral_v<T>>> {
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
        using PromotedT = decltype(+std::declval<U>());
        static_assert(!std::is_same_v<PromotedT, U>);

        PromotedT promoted;
        call(value, promoted);
        target = check_out_of_range<U>(promoted);
    }
};

template <typename T>
struct FromStringImpl<T, std::enable_if_t<std::is_floating_point_v<T>>> {
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
#endif

/**
 * @brief Converts a string to a value.
 *
 * @tparam T The type to convert to.
 * @tparam Char The kind of character.
 * @param v The string to convert.
 *
 * @return The value represented by the input string.
 * @throw bad_lexical_cast Throws if the string could not be converted.
 *
 * @versionadded{1.0.0}
 */
template <typename T, typename Char>
T from_string(std::basic_string<Char> const &v) {
    T target;
    try {
        FromStringImpl<T>::call(v, target);
    } catch (...) {
        return bad_lexical_cast();
    }
    return target;
}

/**
 * @brief Converts a string to a value.
 *
 * @tparam T The type to convert to.
 * @tparam Char The kind of character.
 * @tparam U The type of the default value.
 * @param v The string to convert.
 * @param default_value The default value if the string could not be converted.
 *
 * @return The value represented by the string, or the default value if the string could not be converted.
 *
 * @versionadded{1.0.0}
 */
template <typename T, typename U, typename Char>
T from_string(std::basic_string<Char> const &v, U &&default_value) {
    T target;
    try {
        FromStringImpl<T>::call(v, target);
        return target;
    } catch (...) {
        return std::forward<U>(default_value);
    }
}

#ifndef DOXYGEN
template <typename T>
T from_string(std::string const &v) {
    T target;
    try {
        FromStringImpl<T>::call(v, target);
    } catch (...) {
        throw bad_lexical_cast();
    }
    return target;
}

template <typename T, typename U>
T from_string(std::string const &v, U &&default_value) {
    T target;
    try {
        FromStringImpl<T>::call(v, target);
        return target;
    } catch (...) {
        return std::forward<U>(default_value);
    }
}
#endif

} // namespace einsums