//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

#include <string>

/*
 * The crash handler's options, declared where the handler reads them.
 */

EINSUMS_NAMESPACE_BEGIN(option)

/// Report a fatal exception and write a minidump instead of dying silently.
///
/// Deliberately separate from install-signal-handlers. On Windows the two are
/// different mechanisms - signals never see a memory fault there - and a Python
/// process wants to keep its own faulthandler on the signals while still
/// getting a report out of a hard crash.
inline constinit cl::ConfigOption<bool> CrashHandler = cl::config_flag(
    "einsums:debug:crash-handler", "Install the crash handler that reports a fatal exception and writes a minidump", "Debug", true);

/// Where minidumps go. Empty, the default, means the working directory.
inline constinit cl::ConfigOption<std::string> CrashDumpDir = cl::config_opt<std::string>(
    "einsums:debug:crash-dump-dir", "Directory for crash minidumps (default: the working directory)", "Debug", "", "DIR");

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give the crash handler's options their command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_Debugging_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_Debugging = register_Einsums_Debugging_options();
}

EINSUMS_NAMESPACE_END()
