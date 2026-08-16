//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Runtime/RuntimeConfiguration.hpp>
#include <Einsums/Runtime/ShutdownFunction.hpp>
#include <Einsums/Runtime/StartupFunction.hpp>

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

EINSUMS_NAMESPACE_BEGIN()

namespace detail {
// Default params to initialize with
[[maybe_unused]] static char                     app_name[] = EINSUMS_APPLICATION_STRING;
[[maybe_unused]] static std::vector<std::string> dummy_argv{app_name};
} // namespace detail

/**
 * @struct InitParams
 *
 * Allows users to add functions to run at startup and shutdown.
 *
 * @versionadded{1.0.0}
 */
struct InitParams {
    /**
     * A function to add to the startup list.
     *
     * @versionadded{1.0.0}
     */
    mutable StartupFunctionType startup;

    /**
     * A function  to add to the shutdown list.
     *
     * @versionadded{1.0.0}
     */
    mutable ShutdownFunctionType shutdown;
};

/**
 * \brief Initialize the runtime and start a function.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT int start(std::function<int()> f, std::vector<std::string> const &argv, InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime and start a function.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT int start(std::nullptr_t f, std::vector<std::string> const &argv, InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime and start a function.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT int start(std::function<int(int, char **)> f, std::vector<std::string> &argv, InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime and start a function.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT int start(std::function<int(int, char const *const *)> f, std::vector<std::string> const &argv,
                         InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime and start a function.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT int start(std::function<int(std::vector<std::string> &)> f, std::vector<std::string> &argv,
                         InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime and start a function.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT int start(std::function<int(std::vector<std::string> const &)> f, std::vector<std::string> const &argv,
                         InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime and start a function.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argc The number of command line arguments.
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT int start(std::function<int(int, char **)> f, int argc, char **argv, InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime and start a function.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argc The number of command line arguments.
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT int start(std::function<int(int, char const *const *)> f, int argc, char const *const *argv,
                         InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime and start a function.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argc The number of arguments.
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
template <typename Function>
int start(Function &&f, int argc, char **argv, InitParams const &params = InitParams()) {
    std::vector<std::string> pass_argv(argv, argv + argc);

    return start(std::forward<Function>(f), pass_argv, params);
}

/**
 * \brief Initialize the runtime and start a function.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argc The number of command line arguments.
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
template <typename Function>
int start(Function &&f, int argc, char const *const *argv, InitParams const &params = InitParams()) {
    std::vector<std::string> const pass_argv(argv, argv + argc);

    return start(std::forward<Function>(f), pass_argv, params);
}

/**
 * \brief Initialize the runtime.
 *
 * \param f entry point of the first task on the einsums runtime. f will be passed all non-einsums
 * command line arguments.
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is stopped
 * \post the runtime is running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void initialize(std::function<int(int, char **)> f, std::vector<std::string> &argv, InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void initialize(std::function<int(int, char const *const *)> f, std::vector<std::string> const &argv,
                               InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime.
 *
 * The runtime must be initialized early on, before other Einsums facilities are
 * used. The entry point \p f is executed as if it were `main`, and the
 * initialization parameters are passed on to the initialization routine.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void initialize(std::function<int()> f, std::vector<std::string> const &argv, InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime.
 *
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void initialize(std::nullptr_t, std::vector<std::string> const &argv, InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime.
 *
 * No task is created on the runtime.
 *
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not initialized
 * \post the runtime is running
 *
 * @versionadded{1.0.0}
 */
inline void initialize(std::vector<std::string> const &argv, InitParams const &params = InitParams()) {
    initialize(nullptr, argv, params);
}

/**
 * \brief Initialize the runtime.
 *
 * The runtime must be initialized early on, before other Einsums facilities are
 * used. The entry point \p f is executed as if it were `main`, receiving the
 * command-line arguments not consumed by Einsums, and the initialization
 * parameters are passed on to the initialization routine.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argc The number of arguments.
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void initialize(std::function<int(int, char **)> f, int argc, char **argv, InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime.
 *
 * The runtime must be initialized early on, before other Einsums facilities are
 * used. The entry point \p f is executed as if it were `main`, receiving the
 * command-line arguments not consumed by Einsums, and the initialization
 * parameters are passed on to the initialization routine.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argc The number of arguments.
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void initialize(std::function<int(int, char const *const *)> f, int argc, char const *const *argv,
                               InitParams const &params = InitParams());
/**
 * \brief Initialize the runtime.
 *
 * The runtime must be initialized early on, before other Einsums facilities are
 * used. Since the entry point is the null pointer, no function is called and the
 * initialization routine exits once it is finished.
 *
 * \param argc The number of arguments.
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void initialize(std::nullptr_t, int argc, char const *const *argv, InitParams const &params = InitParams());

/**
 * \brief Initialize the runtime.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argc The number of arguments.
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
template <typename Function>
    requires(!std::is_same_v<Function, std::nullptr_t>)
void initialize(Function &&f, int argc, char **argv, InitParams const &params = InitParams()) {
    std::vector<std::string> pass_argv(argv, argv + argc);

    initialize(f, pass_argv, params);
}

/**
 * \brief Initialize the runtime.
 *
 * \param f entry point of the first task on the einsums runtime
 * \param argc The number of arguments.
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
template <typename Function>
    requires(!std::is_same_v<Function, std::nullptr_t>)
void initialize(Function &&f, int argc, char const *const *argv, InitParams const &params = InitParams()) {
    std::vector<std::string> const pass_argv(argv, argv + argc);

    initialize(f, pass_argv, params);
}

/**
 * \brief Initialize the runtime.
 *
 * \param argc Then number of arguments.
 * \param argv array of arguments. The first element is ignored.
 * \param params The initialization parameters.
 *
 * \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
 * \pre the runtime is not running
 *
 * @versionadded{1.0.0}
 */
inline void initialize(int argc, char const *const *argv, InitParams const &params = InitParams()) {
    initialize(nullptr, argc, argv, params);
}

/**
 * \brief Signal the runtime that it may be stopped.
 *
 * \pre the runtime is initialized
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT int finalize();

EINSUMS_NAMESPACE_END()