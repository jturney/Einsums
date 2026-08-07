//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN()
/**
 * Figures out the characters in the first string that are not in the second.
 *
 * @param[in] st1 The minuend.
 * @param[in] st2 The subtrahend.
 *
 * @return A string containing the characters in the first string that are not in the second.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT std::string difference(std::string const &st1, std::string const &st2);

/**
 * Reverses a string.
 *
 * @param[in] str The string to reverse.
 *
 * @return The input string reversed.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT std::string reverse(std::string const &str);

/**
 * Go through the first string and find the index of each character in the second string. Put them in the output vector
 * in the order they show up in the first string.
 *
 * @param[in] needles The string with the characters being searched for.
 * @param[in] haystack The string whose characters are being indexed.
 * @param[out] out The vector containing the indices.
 *
 * @versionadded{2.0.0}
 */
template <typename int_type, typename Alloc>
void find_char_with_position(std::string const &needles, std::string const &haystack, std::vector<int_type, Alloc> *out) {
    out->resize(needles.length());

    for (int i = 0; i < needles.size(); i++) {
        out->at(i) = (int_type)haystack.find(needles.at(i));
    }
}

EINSUMS_NAMESPACE_END()