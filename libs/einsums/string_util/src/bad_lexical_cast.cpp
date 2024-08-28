//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/string_util/bad_lexical_cast.hpp>

#include <typeinfo>

namespace einsums::string_util::detail {

const char *bad_lexical_cast::what() const noexcept {
    return "bad lexical cast: source type value could not be interpreted as target";
}

bad_lexical_cast::~bad_lexical_cast() noexcept = default;

void throw_bad_lexical_cast(std::type_info const &source_type, std::type_info const &target_type) {
    throw bad_lexical_cast(source_type, target_type);
}

} // namespace einsums::string_util::detail