//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cctype>
#include <string>

namespace einsums::string_util {
template <typename CharT, typename Traits, typename Allocator>
struct is_any_of_pred {
    bool operator()(int c) const noexcept { return chars.find(c) != std::string::npos; }

    std::basic_string<CharT, Traits, Allocator> chars;
};

template <typename CharT, typename Traits, typename Allocator>
is_any_of_pred<CharT, Traits, Allocator> is_any_of(std::basic_string<CharT, Traits, Allocator> const &chars) {
    return is_any_of_pred<CharT, Traits, Allocator>{chars};
}

inline auto is_any_of(char const *chars) {
    return is_any_of_pred<char, std::char_traits<char>, std::allocator<char>>{std::string{chars}};
}

} // namespace einsums::string_util
