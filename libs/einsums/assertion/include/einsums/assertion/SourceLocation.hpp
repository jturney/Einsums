//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/ExportDefinitions.hpp>
#include <iosfwd>

namespace einsums::detail {
/// This contains the location information where \a EINSUMS_ASSERT has been
/// called
struct SourceLocation {
    const char *file_name;
    unsigned    line_number;
    const char *function_name;
};
EINSUMS_EXPORT std::ostream &operator<<(std::ostream &os, SourceLocation const &loc);

} // namespace einsums::detail