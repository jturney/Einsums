//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/export_definitions.hpp>

#include <iosfwd>

namespace einsums::detail {
/// This contains the location information where \a EINSUMS_ASSERT has been
/// called
struct source_location {
    char const *file_name;
    unsigned    line_number;
    char const *function_name;
};
EINSUMS_EXPORT auto operator<<(std::ostream &os, source_location const &loc) -> std::ostream &;

} // namespace einsums::detail