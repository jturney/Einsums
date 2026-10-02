//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <string>

EINSUMS_NAMESPACE_BEGIN(util)

/**
 * @brief Install a last-chance handler that reports a hard crash instead of dying silently.
 *
 * On Windows, registers a @c SetUnhandledExceptionFilter; nothing else in einsums sees an access
 * violation there, so a faulting process would otherwise die with no output. It writes the
 * exception code and address, a backtrace when @c EINSUMS_WITH_BACKTRACES is on, and a minidump
 * to stderr and @p dump_directory, then returns @c EXCEPTION_CONTINUE_SEARCH so the platform's
 * own error reporting still runs.
 *
 * Independent of @c install-signal-handlers, so a Python process can leave @c SIGSEGV to its
 * @c faulthandler and still get a report. Idempotent; a no-op elsewhere, where the Runtime
 * module's signal handlers report crashes.
 *
 * @param dump_directory Where to write the minidump. Empty means the working directory.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void install_crash_handler(std::string const &dump_directory = {});

/**
 * @brief Remove a handler installed by @ref install_crash_handler.
 *
 * Restores the previous filter. Safe to call when none was installed.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void remove_crash_handler();

EINSUMS_NAMESPACE_END(util)
