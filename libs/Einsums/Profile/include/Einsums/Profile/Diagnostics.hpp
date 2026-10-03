//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <cstdint>
#    include <functional>
#    include <string_view>

EINSUMS_NAMESPACE_BEGIN(profile)

/// How much a profiler message matters.
enum class DiagnosticLevel : std::uint8_t {
    Debug,
    Info,
    Warning,
    Error,
};

/// Receives the profiler's own messages: a server that could not bind, a setting refused.
using DiagnosticHandler = std::function<void(DiagnosticLevel level, std::string_view message)>;

/**
 * @brief Send the profiler's messages to @p handler instead of the default.
 *
 * The default writes warnings and errors to stderr and drops the rest. A host with a logger of its
 * own installs a handler that forwards to it. An empty handler restores the default. Thread-safe.
 */
EINSUMS_EXPORT void set_diagnostic_handler(DiagnosticHandler handler);

/// Report @p message at @p level through the installed handler. Thread-safe.
EINSUMS_EXPORT void diagnostic(DiagnosticLevel level, std::string_view message);

EINSUMS_NAMESPACE_END(profile)

#endif
