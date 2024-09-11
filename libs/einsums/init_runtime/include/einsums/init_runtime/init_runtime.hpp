//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/modules/program_options.hpp>
#include <einsums/preprocessor/stringize.hpp>
#include <einsums/runtime/runtime.hpp>
#include <einsums/runtime/shutdown_function.hpp>
#include <einsums/runtime/startup_function.hpp>

#include <csignal>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(EINSUMS_APPLICATION_NAME_DEFAULT) && !defined(EINSUMS_APPLICATION_NAME)
#    define EINSUMS_APPLICATION_NAME EINSUMS_APPLICATION_NAME_DEFAULT
#endif

#if !defined(EINSUMS_APPLICATION_STRING)
#    if defined(EINSUMS_APPLICATION_NAME)
#        define EINSUMS_APPLICATION_STRING EINSUMS_PP_STRINGIZE(EINSUMS_APPLICATION_NAME)
#    else
#        define EINSUMS_APPLICATION_STRING "unknown einsums application"
#    endif
#endif

namespace einsums {
namespace detail {

// Default parameters to initialize the init_params struct
EINSUMS_MAYBE_UNUSED static int    dummy_argc      = 1;
EINSUMS_MAYBE_UNUSED static char   app_name[]      = EINSUMS_APPLICATION_STRING;
static char                       *default_argv[2] = {app_name, nullptr};
EINSUMS_MAYBE_UNUSED static char **dummy_argv      = default_argv;

// EINSUMS_APPLICATION_STRING is specific to an application and therefore
// cannot be in the source file
EINSUMS_MAYBE_UNUSED static const einsums::program_options::options_description default_desc =
    einsums::program_options::options_description("Usage: " EINSUMS_APPLICATION_STRING " [options]");

} // namespace detail

struct init_params {
    std::reference_wrapper<einsums::program_options::options_description const> desc_cmdline = detail::default_desc;
    std::vector<std::string>                                                    cfg;
    mutable startup_function_type                                               startup;
    mutable shutdown_function_type                                              shutdown;
};

EINSUMS_EXPORT int init(std::function<int(einsums::program_options::variables_map &)> f, int argc, const char *const *argv,
                        init_params const &params = init_params());
EINSUMS_EXPORT int init(std::function<int(int, char **)> f, int argc, const char *const *argv, init_params const &params = init_params());
EINSUMS_EXPORT int init(std::function<int()> f, int argc, const char *const *argv, init_params const &params = init_params());
EINSUMS_EXPORT int init(std::nullptr_t, int argc, const char *const *argv, init_params const &params = init_params());

/// Start the runtime.
///
/// @param f entry point of the first task on the einsums runtime. f will be passed all non-einsums
/// command line arguments.
/// @param argc number of arguments in argv
/// @param argv array of arguments. The first element is ignored.
///
/// @pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// @pre the runtime is stopped
/// @post the runtime is running
EINSUMS_EXPORT void start(std::function<int(einsums::program_options::variables_map &)> f, int argc, const char *const *argv,
                          init_params const &params = init_params());

/// Start the runtime.
///
/// @param f entry point of the first task on the einsums runtime. f will be passed all non-einsums
/// command line arguments.
/// @param argc number of arguments in argv
/// @param argv array of arguments. The first element is ignored.
///
/// @pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// @pre the runtime is stopped
/// @post the runtime is running
EINSUMS_EXPORT void start(std::function<int(int, char **)> f, int argc, const char *const *argv, init_params const &params = init_params());

/// Start the runtime.
///
/// @param f entry point of the first task on the einsums runtime
/// @param argc number of arguments in argv
/// @param argv array of arguments. The first element is ignored.
///
/// @pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// @pre the runtime is not running
EINSUMS_EXPORT void start(std::function<int()> f, int argc, const char *const *argv, init_params const &params = init_params());

EINSUMS_EXPORT void start(std::nullptr_t, int argc, const char *const *argv, init_params const &params = init_params());

/// Start the runtime.
///
/// No task is created on the runtime.
///
/// @param argc number of arguments in argv
/// @param argv array of arguments. The first element is ignored.
///
/// @pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// @pre the runtime is not initialized
/// @post the runtime is running
EINSUMS_EXPORT void start(int argc, const char *const *argv, init_params const &params = init_params());

/// Stop the runtime.
///
/// Waits until @ref einsums::finalize has been called and there is no more activity on the
/// runtime. See @ref einsums::wait. The runtime can be started again after calling @ref
/// einsums::stop. Must be called from outside the runtime.
///
/// @return the return value of the callable passed to @p einsums::start, if any. If none was
/// passed, returns 0.
///
/// @pre the runtime is initialized
/// @pre the calling thread is not a einsums task
/// @post the runtime is not initialized
EINSUMS_EXPORT int stop();

/// Signal the runtime that it may be stopped.
///
/// Until @ref einsums::finalize has been called, @ref einsums::stop will not return. This function
/// exists to distinguish between the runtime being idle but still expecting work to be
/// scheduled on it and the runtime being idle and ready to be shutdown. Unlike @einsums::stop,
/// @ref einsums::finalize can be called from within or outside the runtime.
///
/// @pre the runtime is initialized
EINSUMS_EXPORT void finalize();

/// Wait for the runtime to be idle.
///
/// Waits until the runtime is idle. This includes tasks scheduled on the thread pools as well
/// as non-tasks such as CUDA kernels submitted through einsums facilities. Can be called from
/// within the runtime, in which case the calling task is ignored when determining idleness.
///
/// @pre the runtime is initialized
/// @post all work submitted before the call to wait is completed
EINSUMS_EXPORT void wait();

/// Suspend the runtime.
///
/// Waits until the runtime is idle and suspends worker threads on all thread pools. Work can be
/// scheduled on the runtime even when it is suspended, but no progress will be made.
///
/// @pre the calling thread is not a einsums task
/// @pre runtime is running or suspended
/// @post runtime is suspended
EINSUMS_EXPORT void suspend();

/// Resume the runtime.
///
/// Resumes the runtime by waking all worker threads on all thread pools.
///
/// @pre the calling thread is not a einsums task
/// @pre runtime is suspended or running
/// @post runtime is running
EINSUMS_EXPORT void resume();

} // namespace einsums