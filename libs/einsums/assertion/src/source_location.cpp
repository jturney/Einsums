//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/assertion/source_location.hpp>

#include <ostream>

namespace einsums::detail {
auto operator<<(std::ostream &os, source_location const &loc) -> std::ostream & {
    os << loc.file_name << ":" << loc.line_number << ": " << loc.function_name;
    return os;
}

} // namespace einsums::detail