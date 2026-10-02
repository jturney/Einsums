//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <string>

EINSUMS_NAMESPACE_BEGIN(string_util)

/**
 * @brief Converts a memory specification string into a number of bytes.
 *
 * A positive number (integer or decimal, with a period or comma) followed by an optional binary
 * prefix (k, m, g, t; any case) and a unit: b or o for bytes, w for words of sizeof(size_t).
 *
 * @versionadded{1.1.0}
 *
 * @throws std::runtime_error If the string is improperly formatted.
 */
EINSUMS_EXPORT size_t memory_string(std::string const &mem_spec);
EINSUMS_NAMESPACE_END(string_util)
