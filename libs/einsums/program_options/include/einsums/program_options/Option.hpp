//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/program_options/Config.hpp>

#include <string>
#include <utility>
#include <vector>

namespace einsums::program_options {

/** Option found in input source.
    Contains a key and a value. The key, in turn, can be a string (name of
    an option), or an integer (position in input source) \-- in case no name
    is specified. The latter is only possible for command line.
    The template parameter specifies the type of char used for storing the
    option's value.
*/
template <typename Char>
struct BasicOption {
    /** String key of this option. Intentionally independent of the template parameters. */
    std::string string_key;
    /** Position key of this option. All options without an explicit name are
    sequentially numbered starting from 0. If an option has explicit name,
    'position_key' is equal to -1. It is possible that both
    position_key and string_key is specified, in case name is implicitly
    added. */
    int position_key;
    /** Option's value */
    std::vector<std::basic_string<Char>> value;
    /** The original unchanged tokens this option was
        created from. */
    std::vector<std::basic_string<Char>> original_tokens;
    /** True if option was not recognized. In that case,
        'string_key' and 'value' are results of purely
        syntactic parsing of source. The original tokens can be
        recovered from the "original_tokens" member.
    */
    bool unregistered;
    /** True if string_key has to be handled
        case insensitive.
    */
    bool case_insensitive;

    BasicOption() : position_key(-1), unregistered(false), case_insensitive(false) {}
    BasicOption(std::string xstring_key, std::vector<std::string> const &xvalue)
        : string_key(std::move(xstring_key)), position_key(-1), value(xvalue), unregistered(false), case_insensitive(false) {}
};

using Option  = BasicOption<char>;
using WOption = BasicOption<wchar_t>;

} // namespace einsums::program_options