//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/forward.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <utility>

namespace einsums::detail {
template <typename It, typename CharT, typename Traits, typename Allocator>
auto substr(std::basic_string<CharT, Traits, Allocator> const &s, It const &first, It const &last)
    -> std::basic_string<CharT, Traits, Allocator> {
    std::size_t const pos   = std::distance(std::begin(s), first);
    std::size_t const count = std::distance(first, last);
    return s.substr(pos, count);
}

enum class token_compress_mode { off, on };

template <typename Container, typename Predicate, typename CharT, typename Traits, typename Allocator>
void split(Container &container, std::basic_string<CharT, Traits, Allocator> const &str, Predicate &&pred,
           token_compress_mode compress_mode = token_compress_mode::off) {
    container.clear();

    auto token_begin = std::begin(str);
    auto token_end   = std::end(str);

    do {
        token_end = std::find_if(token_begin, std::end(str), pred);

        container.push_back(substr(str, token_begin, token_end));

        if (token_end != std::end(str)) {
            token_begin = token_end + 1;
        }

        if (compress_mode == token_compress_mode::on) {
            // Skip contiguous separators
            while (token_begin != std::end(str) && pred(int(*token_begin))) {
                ++token_begin;
            }
        }
    } while (token_end != std::end(str));
}

template <typename Container, typename Predicate>
void split(Container &container, char const *str, Predicate &&pred,
           token_compress_mode compress_mode = token_compress_mode::off) {
    split(container, std::string{str}, EINSUMS_FORWARD(Predicate, pred), compress_mode);
}
} // namespace einsums::detail
