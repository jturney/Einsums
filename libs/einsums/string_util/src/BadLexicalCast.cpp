//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/string_util/BadLexicalCast.hpp>

namespace einsums::detail {

const char *BadLexicalCast::what() const noexcept {
    return "BadLexicalCast: source type value could not be interpreted as target";
}

BadLexicalCast::~BadLexicalCast() noexcept = default;

void throw_bad_lexical_cast(std::type_info const &source_type, std::type_info const &target_type) {
    throw BadLexicalCast(source_type, target_type);
}

} // namespace einsums::detail