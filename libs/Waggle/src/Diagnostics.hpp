//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Waggle/Config.hpp>

#include <Waggle/Types.hpp>

#include <cstdint>
#include <functional>
#include <string_view>

WAGGLE_NAMESPACE_BEGIN

/**
 * @brief Send the profiler's messages to @p handler instead of the default.
 *
 * The default writes warnings and errors to stderr and drops the rest. A host with a logger of its
 * own installs a handler that forwards to it. An empty handler restores the default. Thread-safe.
 */
WAGGLE_EXPORT void install_diagnostic_handler(DiagnosticHandler handler);

/// Report @p message at @p level through the installed handler. Thread-safe.
WAGGLE_EXPORT void diagnostic(DiagnosticLevel level, std::string_view message);

WAGGLE_NAMESPACE_END
