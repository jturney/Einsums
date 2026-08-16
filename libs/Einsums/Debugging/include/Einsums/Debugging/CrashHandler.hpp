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
 * On Windows this registers a @c SetUnhandledExceptionFilter. Nothing else in einsums
 * catches an access violation there: @ref set_signal_handlers installs only a
 * @c SetConsoleCtrlHandler, which sees Ctrl-C and session shutdown and never sees a
 * memory fault. Without this a faulting process is killed by the OS with no output at
 * all, which is how a crash reaches a log as a bare "SEGFAULT" and nothing more.
 *
 * The handler writes, to stderr, the exception code and faulting address, then a
 * backtrace when @c EINSUMS_WITH_BACKTRACES is on, and finally a minidump next to the
 * working directory. Writing the dump needs no elevation and no registry
 * configuration: a process may always dump itself.
 *
 * The filter returns @c EXCEPTION_CONTINUE_SEARCH so the platform's own error
 * reporting still runs afterwards. That composes rather than competes: a machine
 * configured to collect dumps out of process (a CI runner, say) collects one as well,
 * which matters because an in-process dump is the less reliable of the two exactly
 * when the fault happens during teardown under the loader lock.
 *
 * Independent of @c install-signal-handlers on purpose. A Python process wants
 * einsums to keep its hands off @c SIGSEGV, so that its own @c faulthandler stays in
 * charge, while still wanting a Windows crash to say something before it dies.
 *
 * Idempotent; calling it twice installs one handler. A no-op on non-Windows
 * platforms, where the signal handlers in the Runtime module carry the diagnostics.
 *
 * @param dump_directory Where to write the minidump. Empty means the working directory.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void install_crash_handler(std::string const &dump_directory = {});

/**
 * @brief Remove a handler installed by @ref install_crash_handler.
 *
 * Restores whatever filter was in place beforehand. Safe to call when none was
 * installed.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void remove_crash_handler();

EINSUMS_NAMESPACE_END(util)
