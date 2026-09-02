//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

/*
 * The runtime's own options, declared where the runtime reads them.
 *
 * Each descriptor is the single place its name, help, type, and default are
 * spelled; the config key and the environment variable derive from the name,
 * and a reader names the descriptor rather than a string.
 */

EINSUMS_NAMESPACE_BEGIN(option)

/// Install the handlers that report a fatal signal before the process dies.
inline constinit cl::ConfigOption<bool> InstallSignalHandlers = cl::config_flag(
    "einsums:debug:install-signal-handlers", "Install signal handlers that report a fatal signal before aborting", "Debug", true);

/// Offer to attach a debugger when a detected error terminates the process.
inline constinit cl::ConfigOption<bool> AttachDebugger =
    cl::config_flag("einsums:debug:attach-debugger", "Provide a mechanism to attach a debugger on detected errors", "Debug", false);

/// Print extra diagnostics on the way down.
inline constinit cl::ConfigOption<bool> DiagnosticsOnTerminate =
    cl::config_flag("einsums:debug:diagnostics-on-terminate", "Print additional diagnostic information on termination", "Debug", true);

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give the runtime's options their command-line presence.
 *
 * Idempotent, and run from a namespace-scope initializer below so that
 * including this header is enough: an option a reader can name is an option
 * `--help` lists.
 */
EINSUMS_EXPORT int register_Einsums_Runtime_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_Runtime = register_Einsums_Runtime_options();
}

EINSUMS_NAMESPACE_END()
