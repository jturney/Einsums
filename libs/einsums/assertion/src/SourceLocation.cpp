//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/assertion/SourceLocation.hpp>
#include <ostream>

namespace einsums::detail {
std::ostream &operator<<(std::ostream &os, SourceLocation const &loc) {
    os << loc.file_name << ":" << loc.line_number << ": " << loc.function_name;
    return os;
}

} // namespace einsums::detail