//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>

#include <string>

namespace einsums::string_util {

/**
 * @brief Converts a memory specification string into a number of bytes.
 *
 * A memory specification has a number followed by a unit. The number must be positive,
 * but it can be either integral or decimal. The unit is made up of a prefix followed by a unit.
 * The case of the prefix does not matter
 * The prefix is either nothing, 'k', 'm', 'g', or 't',
 * representing a binary kilo (1024), mega (1024 k), giga (1024 M),
 * or tera (1024 G). After the prefix, if 'b', 'B', 'o', or 'O' are given,
 * then the number will be treated as a number of bytes. If 'w' or 'W' are given,
 * then the number will be treated as a number of words. The size of a word is
 * platform dependent, but is considered to be the size of @c size_t , which is
 * usually 8 bytes on 64-bit systems.
 * Both comma and period decimals are recognized.
 *
 * @versionadded{1.1.0}
 *
 * @throws std::runtime_error If the string is improperly formatted.
 */
EINSUMS_EXPORT size_t memory_string(std::string const &mem_spec);
} // namespace einsums::string_util
