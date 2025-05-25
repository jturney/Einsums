//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile/Timer.hpp>
#include <Einsums/Runtime/Detail/InitLogging.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Runtime/Runtime.hpp>
#include <Einsums/Utilities/Random.hpp>
#include <Einsums/Version.hpp>

#include <csignal>
#include <cstdlib>
#include <functional>
#include <h5cpp/all>
#include <spdlog/spdlog.h>
#include <tuple>
#include <unordered_map>

namespace einsums {

template <typename F, typename... BoundArgs>
struct bind_back_t {
    F                        func_;
    std::tuple<BoundArgs...> bound_args_;

    template <typename... CallArgs, std::size_t... Indices>
    auto invoke_impl(std::index_sequence<Indices...>, CallArgs &&...call_args) const {
        return func_(std::forward<CallArgs>(call_args)..., std::get<Indices>(bound_args_)...);
    }

    template <typename... CallArgs, std::size_t... Indices>
    auto invoke_impl(std::index_sequence<Indices...>, CallArgs &&...call_args) {
        return func_(std::forward<CallArgs>(call_args)..., std::get<Indices>(bound_args_)...);
    }

    template <typename Func, typename... Args>
    bind_back_t(Func &&func, Args &&...args) : func_(std::forward<Func>(func)), bound_args_(std::forward<Args>(args)...) {}

    template <typename... CallArgs>
    auto operator()(CallArgs &&...call_args) const {
        return invoke_impl(std::index_sequence_for<BoundArgs...>{}, std::forward<CallArgs>(call_args)...);
    }

    template <typename... CallArgs>
    auto operator()(CallArgs &&...call_args) {
        return invoke_impl(std::index_sequence_for<BoundArgs...>{}, std::forward<CallArgs>(call_args)...);
    }
};

template <typename F, typename... BoundArgs>
auto bind_back(F &&func, BoundArgs &&...bound_args) {
    return bind_back_t<std::decay_t<F>, std::decay_t<BoundArgs>...>(std::forward<F>(func), std::forward<BoundArgs>(bound_args)...);
}

namespace detail {

void add_startup_functions(Runtime &rt, RuntimeConfiguration const &cfg, StartupFunctionType startup, ShutdownFunctionType shutdown) {
    if (!!startup) {
        rt.add_startup_function(std::move(startup));
    }

    if (!!shutdown) {
        rt.add_shutdown_function(std::move(shutdown));
    }
}

int run(std::function<int()> const &f, Runtime &rt, InitParams const &params) {
    add_startup_functions(rt, rt.config(), std::move(params.startup), std::move(params.shutdown));

    // Run this runtime instance using the given function f
    if (f) {
        return rt.run(f);
    }

    // Run this runtime instance without an einsums_main
    return rt.run();
}

int run(std::function<int()> const &f, std::vector<std::string> const &argv, InitParams const &params, bool blocking) {
    // EINSUMS_LOG_INFO("Running common initialization routines...");
    /// @todo Add a check to ensure the runtime hasn't already been initialized

    // Command line arguments for Einsums will be prefixed with --einsums:
    // For example, "--einsums:verbose=1" will be translated to verbose=1
    RuntimeConfiguration config(argv, params.register_arguments);

    // Before this line logging does not work.
    init_logging(config);

    auto &global_config = GlobalConfigMap::get_singleton();

    // Report build settings.
    EINSUMS_LOG_INFO("Starting Einsums: {}", build_string());

    if (global_config.get_bool("install-signal-handlers")) {
        EINSUMS_LOG_TRACE("Installing signal handlers...");
        set_signal_handlers();
    }

    // This is the only initialization routine that needs to be explicitly called here.
    // This is because the runtime environment depends on the profiler. If the profiler
    // depended on the runtime environment, then there would be a dependency issue.
    profile::initialize();

    // Disable HDF5 diagnostic reporting
    H5Eset_auto(0, nullptr, nullptr);

    // Build and configure this runtime instance.
    std::unique_ptr<Runtime> rt = std::make_unique<Runtime>(std::move(config), true);

    if (blocking) {
        return run(f, *rt, params);
    }

    run(f, *rt, params);

    // pointer to runtime is stored in TLS
    Runtime *p = rt.release();

    detail::register_free_pointer([p]() { delete p; });

    return 0;
}

int run_impl(std::function<int()> f, std::vector<std::string> const &argv, InitParams const &params, bool blocking) {
    std::vector<std::string> const *pass_argv = &argv;
    if (argv.size() == 0) {
        pass_argv = &dummy_argv;
    }

    // register default handlers
    [[maybe_unused]] auto signal_handler = std::signal(SIGABRT, on_abort);
    [[maybe_unused]] auto exit_result    = std::atexit(on_exit);
#if defined(EINSUMS_HAVE_CXX11_STD_QUICK_EXIT)
    [[maybe_unused]] auto quick_exit_result = std::at_quick_exit(on_exit);
#endif
    return run(f, *pass_argv, params, blocking);
}

} // namespace detail

int start(std::function<int()> f, std::vector<std::string> const &argv, InitParams const &params) {
    return detail::run_impl(std::move(f), argv, params, true);
}

int start(std::nullptr_t, std::vector<std::string> const &argv, InitParams const &params) {
    std::function<int()> main_f;
    return detail::run_impl(std::move(main_f), argv, params, true);
}

int start(std::function<int(int, char **)> f, std::vector<std::string> &argv, InitParams const &params) {
    std::vector<char *> copy_argv(argv.size()); // We do it this way so that the memory gets freed on return.

    for (ptrdiff_t i = 0; i < argv.size(); i++) {
        /*BADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODE
         *BADCODE                                                                             BADCODE
         *BADCODE                              BAD CODE ALERT                                 BADCODE
         *BADCODE                                                                             BADCODE
         *BADCODE   ATTENTION: THIS IS BAD CODE. IT WILL NEED TO BE REWRITTEN IN THE FUTURE.  BADCODE
         *BADCODE         MEMORY SAFETY IS NOT ONLY NOT GUARANTEED BUT OUTRIGHT FLOUTED.      BADCODE
         *BADCODE         WHEN REWRITING, PLEASE ENSURE THAT THE MEMORY IS BOTH SAFE ON       BADCODE
         *BADCODE         ENTRY TO THE CALL OF THE MAIN FUNCTION AND PROPERLY DESTROYED       BADCODE
         *BADCODE         ON EXIT. THIS IS A TEMPORARY FIX ONLY.                              BADCODE
         *BADCODE                                                                             BADCODE
         *BADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODE
         */
        /// @todo Fix bad code.
        copy_argv[i] = const_cast<char *>(argv[i].c_str());
    }

    std::function<int()> main_f = std::bind(f, (int)copy_argv.size(), copy_argv.data());

    return detail::run_impl(main_f, argv, params, true);
}

int start(std::function<int(int, char const *const *)> f, std::vector<std::string> const &argv, InitParams const &params) {
    std::vector<char const *> copy_argv(argv.size()); // We do it this way so that the memory gets freed on return.

    for (ptrdiff_t i = 0; i < argv.size(); i++) {
        copy_argv[i] = argv[i].c_str();
    }

    std::function<int()> main_f = std::bind(f, (int)copy_argv.size(), copy_argv.data());

    return detail::run_impl(main_f, argv, params, true);
}

int start(std::function<int(std::vector<std::string> &)> f, std::vector<std::string> &argv, InitParams const &params) {
    std::function<int()> main_f = std::bind(f, argv);

    return detail::run_impl(main_f, argv, params, true);
}

int start(std::function<int(std::vector<std::string> const &)> f, std::vector<std::string> const &argv, InitParams const &params) {
    std::function<int()> main_f = std::bind(f, argv);

    return detail::run_impl(main_f, argv, params, true);
}

int start(std::function<int(int, char **)> f, int argc, char **argv, InitParams const &params) {
    std::vector<std::string> pass_argv(argv, argv + argc);

    std::function<int()> main_f = std::bind(f, argc, argv);

    return detail::run_impl(main_f, pass_argv, params, true);
}

int start(std::function<int(int, char const *const *)> f, int argc, char const *const *argv, InitParams const &params) {
    std::vector<std::string> pass_argv(argv, argv + argc);

    std::function<int()> main_f = std::bind(f, argc, argv);

    return detail::run_impl(main_f, pass_argv, params, true);
}

void initialize(std::function<int(int, char **)> f, std::vector<std::string> &argv, InitParams const &params) {
    std::vector<char *> copy_argv(argv.size()); // We do it this way so that the memory gets freed on return.

    for (ptrdiff_t i = 0; i < argv.size(); i++) {
        /*BADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODE
         *BADCODE                                                                             BADCODE
         *BADCODE                              BAD CODE ALERT                                 BADCODE
         *BADCODE                                                                             BADCODE
         *BADCODE   ATTENTION: THIS IS BAD CODE. IT WILL NEED TO BE REWRITTEN IN THE FUTURE.  BADCODE
         *BADCODE         MEMORY SAFETY IS NOT ONLY NOT GUARANTEED BUT OUTRIGHT FLOUTED.      BADCODE
         *BADCODE         WHEN REWRITING, PLEASE ENSURE THAT THE MEMORY IS BOTH SAFE ON       BADCODE
         *BADCODE         ENTRY TO THE CALL OF THE MAIN FUNCTION AND PROPERLY DESTROYED       BADCODE
         *BADCODE         ON EXIT. THIS IS A TEMPORARY FIX ONLY.                              BADCODE
         *BADCODE                                                                             BADCODE
         *BADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODEBADCODE
         */
        /// @todo Fix bad code.
        copy_argv[i] = const_cast<char *>(argv[i].c_str());
    }

    std::function<int()> main_f = std::bind(f, (int)copy_argv.size(), copy_argv.data());

    if (detail::run_impl(std::move(main_f), argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void initialize(std::function<int(int, char const *const *)> f, std::vector<std::string> const &argv, InitParams const &params) {
    std::vector<char const *> copy_argv(argv.size()); // We do it this way so that the memory gets freed on return.

    for (ptrdiff_t i = 0; i < argv.size(); i++) {
        copy_argv[i] = argv[i].c_str();
    }

    std::function<int()> main_f = std::bind(f, (int)copy_argv.size(), copy_argv.data());

    if (detail::run_impl(std::move(main_f), argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void initialize(std::function<int()> f, std::vector<std::string> const &argv, InitParams const &params) {
    if (detail::run_impl(std::move(f), argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void initialize(std::nullptr_t, std::vector<std::string> const &argv, InitParams const &params) {
    std::function<int()> main_f;
    if (detail::run_impl(std::move(main_f), argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void initialize(std::function<int(int, char **)> f, int argc, char **argv, InitParams const &params) {
    std::vector<std::string> pass_argv(argv, argv + argc);
    std::function<int()>     main_f = std::bind(f, argc, argv);
    if (detail::run_impl(std::move(main_f), pass_argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void initialize(std::function<int(int, char const *const *)> f, int argc, char const *const *argv, InitParams const &params) {
    std::vector<std::string> pass_argv(argv, argv + argc);
    std::function<int()>     main_f = std::bind(f, argc, argv);
    if (detail::run_impl(std::move(main_f), pass_argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void initialize(std::nullptr_t, int argc, char const *const *argv, InitParams const &params) {
    std::function<int()> main_f;

    initialize(main_f, argc, argv, params);
}

} // namespace einsums