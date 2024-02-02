//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace einsums::detail {

template <typename CharT, typename Traits, typename Alloc>
void to_lower(std::basic_string<CharT, Traits, Alloc> &s) {
    std::transform(std::begin(s), std::end(s), std::begin(s), [](int c) { return std::tolower(c); });
}

template <typename CharT, typename Traits, typename Alloc>
void to_upper(std::basic_string<CharT, Traits, Alloc> &s) {
    std::transform(std::begin(s), std::end(s), std::begin(s), [](int c) { return std::toupper(c); });
}

} // namespace einsums::detail