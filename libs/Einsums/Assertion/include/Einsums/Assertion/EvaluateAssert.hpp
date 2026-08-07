//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <source_location>
#include <string>

EINSUMS_NAMESPACE_BEGIN(detail)

/**
 * @brief Handles assertion behavior, selecting between a user-defined handler or the default handler.
 *
 * @param[in] loc The source location for constructing a message.
 * @param[in] expr A string representing the condition being evaluated.
 * @param[in] msg An extra message to print out.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      No longer noexcept to allow users to raise an exception rather than immediately calling exit.
 * @endversion
 */
EINSUMS_EXPORT void handle_assert(std::source_location const &loc, char const *expr, std::string const &msg);

EINSUMS_NAMESPACE_END(detail)