//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>
#include <Einsums/RuntimeConfiguration/ShutdownFunction.hpp>
#include <Einsums/RuntimeConfiguration/StartupFunction.hpp>

#include <functional>
#include <string>
#include <type_traits>

#if defined(EINSUMS_APPLICATION_NAME_DEFAULT) && !defined(EINSUMS_APPLICATION_NAME)
#    define EINSUMS_APPLICATION_NAME EINSUMS_APPLICATION_NAME_DEFAULT
#endif

#if !defined(EINSUMS_APPLICATION_STRING)
#    if defined(EINSUMS_APPLICAITON_NAME)
#        define EINSUMS_APPLICATION_STRING EINSUMS_PP_STRING(EINSUMS_APPLICATION_NAME)
#    else
#        define EINSUMS_APPLICATION_STRING "unknown Einsums application"
#    endif
#endif

namespace einsums {

namespace detail {
// Default params to initialize with
[[maybe_unused]] static char                     app_name[] = EINSUMS_APPLICATION_STRING;
[[maybe_unused]] static std::vector<std::string> dummy_argv{app_name};
} // namespace detail

/**
 * @struct InitParams
 *
 * @todo Document.
 */
struct InitParams {
    mutable StartupFunctionType  startup;
    mutable ShutdownFunctionType shutdown;
};

/// \brief Initialize the runtime and start a function.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT int start(std::function<int()> f, std::vector<std::string> const &argv, InitParams const &params = InitParams());

/// \brief Initialize the runtime and start a function.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT int start(std::nullptr_t f, std::vector<std::string> const &argv, InitParams const &params = InitParams());

/// \brief Initialize the runtime and start a function.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT int start(std::function<int(int, char **)> f, std::vector<std::string> &argv, InitParams const &params = InitParams());

/// \brief Initialize the runtime and start a function.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT int start(std::function<int(int, char const *const *)> f, std::vector<std::string> const &argv,
                         InitParams const &params = InitParams());

/// \brief Initialize the runtime and start a function.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT int start(std::function<int(std::vector<std::string> &)> f, std::vector<std::string> &argv,
                         InitParams const &params = InitParams());

/// \brief Initialize the runtime and start a function.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT int start(std::function<int(std::vector<std::string> const &)> f, std::vector<std::string> const &argv,
                         InitParams const &params = InitParams());

/// \brief Initialize the runtime and start a function.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argc The number of command line arguments.
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT int start(std::function<int(int, char **)> f, int argc, char **argv, InitParams const &params = InitParams());

/// \brief Initialize the runtime and start a function.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argc The number of command line arguments.
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT int start(std::function<int(int, char const *const *)> f, int argc, char const *const *argv,
                         InitParams const &params = InitParams());

/// \brief Initialize the runtime and start a function.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argc The number of arguments.
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
template <typename Function>
int start(Function &&f, int argc, char **argv, InitParams const &params = InitParams()) {
    std::vector<std::string> pass_argv(argv, argv + argc);

    return start(std::forward<Function>(f), pass_argv, params);
}

/// \brief Initialize the runtime and start a function.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argc The number of command line arguments.
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
template <typename Function>
int start(Function &&f, int argc, char const *const *argv, InitParams const &params = InitParams()) {
    std::vector<std::string> const pass_argv(argv, argv + argc);

    return start(std::forward<Function>(f), pass_argv, params);
}

/// \brief Initialize the runtime.
///
/// \param f entry point of the first task on the einsums runtime. f will be passed all non-einsums
/// command line arguments.
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is stopped
/// \post the runtime is running
EINSUMS_EXPORT void initialize(std::function<int(int, char **)> f, std::vector<std::string> &argv, InitParams const &params = InitParams());

/// \brief Initialize the runtime.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT void initialize(std::function<int(int, char const *const *)> f, std::vector<std::string> const &argv,
                               InitParams const &params = InitParams());

/// \brief Initialize the runtime.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT void initialize(std::function<int()> f, std::vector<std::string> const &argv, InitParams const &params = InitParams());

/// \brief Initialize the runtime.
///
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT void initialize(std::nullptr_t, std::vector<std::string> const &argv, InitParams const &params = InitParams());

/// \brief Initialize the runtime.
///
/// No task is created on the runtime.
///
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not initialized
/// \post the runtime is running
inline void initialize(std::vector<std::string> const &argv, InitParams const &params = InitParams()) {
    initialize(nullptr, argv, params);
}
/// \brief Initialize the runtime.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argc The number of arguments.
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT void initialize(std::function<int(int, char **)> f, int argc, char **argv, InitParams const &params = InitParams());

/// \brief Initialize the runtime.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argc The number of arguments.
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT void initialize(std::function<int(int, char const *const *)> f, int argc, char const *const *argv,
                               InitParams const &params = InitParams());

/// \brief Initialize the runtime.
///
/// \param argc The number of arguments.
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT void initialize(std::nullptr_t, int argc, char const *const *argv, InitParams const &params = InitParams());

/// \brief Initialize the runtime.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argc The number of arguments.
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
template <typename Function>
    requires(!std::is_same_v<Function, std::nullptr_t>)
void initialize(Function &&f, int argc, char **argv, InitParams const &params = InitParams()) {
    std::vector<std::string> pass_argv(argv, argv + argc);

    initialize(f, pass_argv, params);
}

/// \brief Initialize the runtime.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argc The number of arguments.
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
template <typename Function>
    requires(!std::is_same_v<Function, std::nullptr_t>)
void initialize(Function &&f, int argc, char const *const *argv, InitParams const &params = InitParams()) {
    std::vector<std::string> const pass_argv(argv, argv + argc);

    initialize(f, pass_argv, params);
}

/// \brief Initialize the runtime.
///
/// \param argc Then number of arguments.
/// \param argv array of arguments. The first element is ignored.
/// \param params The initialization parameters.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
inline void initialize(int argc, char const *const *argv, InitParams const &params = InitParams()) {
    initialize(nullptr, argc, argv, params);
}

/// \brief Signal the runtime that it may be stopped.
///
/// \pre the runtime is initialized
EINSUMS_EXPORT int finalize();

} // namespace einsums