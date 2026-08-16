//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <cctype>
#include <ranges>
#include <string>

EINSUMS_NAMESPACE_BEGIN(string_util)

/**
 * Trim whitespace from the start of the string.
 *
 * @param[inout] s The string to trim.
 *
 * @versionadded{1.0.0}
 */
static inline void ltrim(std::string &s) {
    auto first_nospace = std::ranges::find_if(s, [](unsigned char ch) { return !std::isspace(ch); });
    if (first_nospace != s.cend()) {
        s.erase(s.begin(), first_nospace);
    }
}

/**
 * Trim whitespace from the end of the string.
 *
 * @param[inout] s The string to trim.
 *
 * @versionadded{1.0.0}
 */
static inline void rtrim(std::string &s) {
    s.erase(std::ranges::find_if(std::views::reverse(s), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

/**
 * Trim whitespace from the start and end of the string.
 *
 * @param[inout] s The string to trim.
 *
 * @versionadded{1.0.0}
 */
static inline void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
}

/**
 * Trim whitespace from the start of the string.
 *
 * @param[in] s The string to trim.
 *
 * @return The string without leading whitespace.
 *
 * @versionadded{1.0.0}
 */
static inline auto ltrim_copy(std::string s) -> std::string {
    ltrim(s);
    return s;
}

/**
 * Trim whitespace from the end of the string.
 *
 * @param[in] s The string to trim.
 *
 * @return The string without trailing whitespace.
 *
 * @versionadded{1.0.0}
 */
static inline auto rtrim_copy(std::string s) -> std::string {
    rtrim(s);
    return s;
}

/**
 * Trim whitespace from the start and end of the string.
 *
 * @param[in] s The string to trim.
 *
 * @return The string without leading or trailing whitespace.
 *
 * @versionadded{1.0.0}
 */
static inline auto trim_copy(std::string s) -> std::string {
    trim(s);
    return s;
}

EINSUMS_NAMESPACE_END(string_util)