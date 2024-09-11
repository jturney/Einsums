//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace einsums::string_util {

inline std::vector<std::string_view> tokenize(std::string_view str, char delimiter) {
    std::vector<std::string_view> tokens;
    size_t                        start = 0, end = 0;

    while ((end = str.find(delimiter, start)) != std::string_view::npos) {
        tokens.emplace_back(str.substr(start, end - start));
        start = end + 1;
    }
    tokens.emplace_back(str.substr(start));

    return tokens;
}

/**
 * @brief Tokenizes a string based on a delimiter, handling escape sequences and quoted strings.
 *
 * This function splits the input string into tokens using the provided delimiter, while respecting escape sequences
 * (such as backslashes) and quoted substrings (which are treated as single tokens). The default escape character
 * is the backslash (`\`), and the default quote character is the double quote (`"`).
 *
 * Example:
 * - Input: `one two\ three "four five" six`
 * - Output: `{"one", "two three", "four five", "six"}`
 *
 * @param input The input string to be tokenized.
 * @param escape_char The character used for escaping (default is `\`).
 * @param delimiter The character used to separate tokens (default is a space `' '`).
 * @param quote_char The character used to indicate quoted strings (default is `"`).
 *
 * @return A vector of strings representing the tokens extracted from the input string.
 *         Quoted substrings are returned as single tokens, and escaped characters are preserved as part of the tokens.
 */
inline std::vector<std::string> split_escaped_list(std::string_view input, char escape_char = '\\', char delimiter = ' ',
                                                   char quote_char = '"') {
    std::vector<std::string> result;
    std::string              current_token;
    bool                     inside_quotes = false;
    bool                     escape_next   = false;

    for (char ch : input) {
        if (escape_next) {
            current_token += ch; // Add the escaped character
            escape_next = false;
            continue;
        }

        if (ch == escape_char) {
            escape_next = true; // Set the flag to escape the next character
            continue;
        }

        if (ch == quote_char) {
            inside_quotes = !inside_quotes; // Toggle inside quote mode
            continue;
        }

        if (ch == delimiter && !inside_quotes) {
            if (!current_token.empty()) {
                result.push_back(current_token); // Tokenize
                current_token.clear();
            }
            continue;
        }

        current_token += ch; // Add character to current token
    }

    if (!current_token.empty()) {
        result.push_back(current_token); // Add the final token
    }

    return result;
}

} // namespace einsums::string_util