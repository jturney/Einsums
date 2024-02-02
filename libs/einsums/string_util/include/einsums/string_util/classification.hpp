//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <string>

namespace einsums::detail {

template <typename CharT, typename Traits, typename Allocator>
struct is_any_of_pred {
    auto operator()(int c) const noexcept -> bool { return chars.find(c) != std::string::npos; }

    std::basic_string<CharT, Traits, Allocator> chars;
};

template <typename CharT, typename Traits, typename Allocator>
auto is_any_of(std::basic_string<CharT, Traits, Allocator> const &chars) -> is_any_of_pred<CharT, Traits, Allocator> {
    return is_any_of_pred<CharT, Traits, Allocator>{chars};
}

inline auto is_any_of(char const *chars) {
    return is_any_of_pred<char, std::char_traits<char>, std::allocator<char>>{std::string{chars}};
}

} // namespace einsums::detail