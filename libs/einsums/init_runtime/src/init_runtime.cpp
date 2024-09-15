//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/command_line_handling/command_line_handling.hpp>
#include <einsums/filesystem.hpp>
#include <einsums/init_runtime/detail/init_logging.hpp>
#include <einsums/init_runtime/init_runtime.hpp>
#include <einsums/logging.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/modules/functional.hpp>
#include <einsums/modules/program_options.hpp>
#include <einsums/runtime/config_entry.hpp>
#include <einsums/runtime/custom_exception_info.hpp>
#include <einsums/runtime/runtime.hpp>
#include <einsums/runtime/runtime_fwd.hpp>
#include <einsums/runtime/runtime_handlers.hpp>
#include <einsums/runtime/shutdown_function.hpp>
#include <einsums/runtime/startup_function.hpp>
#include <einsums/runtime_configuration/runtime_configuration.hpp>
#include <einsums/util/get_entry_as.hpp>
#include <einsums/version.hpp>

#include <functional>
#include <iostream>

#if defined(EINSUMS_HAVE_HIP)
#    include <hip/hip_runtime.h>
#endif

#if !defined(EINSUMS_WINDOWS)
#    include <signal.h>
#endif

namespace einsums {
namespace detail {

int init_helper(program_options::variables_map & /*vm*/, std::function<int(int, char **)> const &f) {
    std::string cmdline(einsums::detail::get_config_entry("einsums.reconstructed_cmd_line", ""));

    using namespace einsums::program_options;
#if defined(EINSUMS_WINDOWS)
    std::vector<std::string> args = split_winmain(cmdline);
#else
    std::vector<std::string> args = split_unix(cmdline);
#endif

    // Copy all arguments which are not einsums related to a temporary array
    std::vector<char *> argv(args.size() + 1);
    std::size_t         argcount = 0;
    for (std::size_t i = 0; i != args.size(); i++) {
        if (args[i].find("--einsums:") != 0) {
            argv[argcount++] = const_cast<char *>(args[i].data());
        } else if (args[i].find("positional", 10) == 10) {
            std::string::size_type p = args[i].find_first_of('=');
            if (p != std::string::npos) {
                args[i]          = args[i].substr(p + 1);
                argv[argcount++] = const_cast<char *>(args[i].data());
            }
        }
    }

    argv[argcount] = nullptr;
    return f(static_cast<int>(argcount), argv.data());
}

void activate_global_options(command_line_handling &cmdline) {
    init_logging(cmdline._rtcfg);
}

struct dump_config {
    explicit dump_config(runtime const &rt) : _rt(std::cref(rt)) {}

    void operator()() const {
        std::cout << "Configuration after runtime start:\n";
        std::cout << "----------------------------------\n";
        _rt.get().get_config().dump(0, std::cout);
        std::cout << "----------------------------------\n";
    }

    std::reference_wrapper<runtime const> _rt;
};

void add_startup_functions(runtime &rt, program_options::variables_map &vm, startup_function_type startup,
                           shutdown_function_type shutdown) {
    if (vm.count("einsums:app-config")) {
        std::string config(vm["einsums:app-config"].as<std::string>());
        rt.get_config().load_application_configuration(config.c_str());
    }

    if (startup) {
        rt.add_startup_function(std::move(startup));
    }

    if (shutdown) {
        rt.add_shutdown_function(std::move(shutdown));
    }

    if (vm.count("einsums:dump-config-initial")) {
        std::cout << "Configuration after runtime construction:\n";
        std::cout << "-----------------------------------------\n";
        rt.get_config().dump(0, std::cout);
        std::cout << "-----------------------------------------\n";
    }

    if (vm.count("einsums:dump-config")) {
        rt.add_startup_function(dump_config(rt));
    }
}

int run(runtime &rt, std::function<int(program_options::variables_map &vm)> const &f, program_options::variables_map &vm,
        startup_function_type startup, shutdown_function_type shutdown) {
    add_startup_functions(rt, vm, std::move(startup), std::move(shutdown));

    // Run this runtime instance using the given function.
    if (f) {
        return rt.run(std::bind_front(f, vm));
    }

    // Run this runinstance instance without a einsums_main
    return rt.run();
}

int start(runtime &rt, std::function<int(program_options::variables_map &)> const &f, program_options::variables_map &vm,
          startup_function_type startup, shutdown_function_type shutdown) {
    add_startup_functions(rt, vm, std::move(startup), std::move(shutdown));

    if (f) {
        return rt.start(std::bind_front(f, vm));
    }

    return rt.start();
}

int run_or_start(bool blocking, std::unique_ptr<runtime> rt, command_line_handling &cfg, startup_function_type startup,
                 shutdown_function_type shutdown) {
    if (blocking) {
        return run(*rt, cfg._einsums_main_f, cfg._vm, std::move(startup), std::move(shutdown));
    }

    // non-blocking version
    start(*rt, cfg._einsums_main_f, cfg._vm, std::move(startup), std::move(shutdown));

    [[maybe_unused]] runtime *p = rt.release();

    return 0;
}

void init_environment(command_line_handling &cmdline) {
    EINSUMS_UNUSED(filesystem::initial_path());

    set_assertion_handler(assertion_handler);
    set_get_full_build_string(&einsums::full_build_string);

    if (get_entry_as<bool>(cmdline._rtcfg, "einsums.install_signal_handlers", false)) {
        set_signal_handlers();
    }
}

int run_or_start(std::function<int(program_options::variables_map &vm)> const &f, int argc, const char *const *argv,
                 init_params const &params, bool blocking) {
    if (get_runtime_ptr() != nullptr) {
        EINSUMS_THROW_EXCEPTION(error::invalid_status, "runtime already initialized");
    }

    command_line_handling cmdline(einsums::util::runtime_configuration(argv[0]), params.cfg, f);
    auto                  cmdline_result = cmdline.call(params.desc_cmdline, argc, argv);

    activate_global_options(cmdline);

    init_environment(cmdline);

    switch (cmdline_result) {
    case command_line_handling_result::success:
        break;
    case command_line_handling_result::exit:
        return 0;
    }

    // Perform any other initialization that needs to happen

    EINSUMS_LOG(info, "run_local: create runtime");

    // Build and configure this runtime instance
    std::unique_ptr<runtime> rt;

    EINSUMS_LOG(info, "creating local runtime");
    rt.reset(new runtime(cmdline._rtcfg, true));

    return run_or_start(blocking, std::move(rt), cmdline, std::move(params.startup), std::move(params.shutdown));
}

int init_start_impl(std::function<int(program_options::variables_map &)> f, int argc, const char *const *argv, init_params const &params,
                    bool blocking) {
    if (argc == 0 || argv == nullptr) {
        argc = dummy_argc;
        argv = dummy_argv;
    }

    [[maybe_unused]] auto signal_handler = std::signal(SIGABRT, detail::on_abort);
    [[maybe_unused]] auto exit_handler   = std::atexit(detail::on_exit);
#if defined(EINSUMS_HAVE_CXX11_STD_QUICK_EXIT)
    [[maybe_unused]] auto quick_exit_result = std::at_quick_exit(detail::on_exit);
#endif
    return run_or_start(f, argc, argv, params, blocking);
}

} // namespace detail

int init(std::function<int(einsums::program_options::variables_map &)> f, int argc, const char *const *argv, init_params const &params) {
    return detail::init_start_impl(std::move(f), argc, argv, params, true);
}

int init(std::function<int(int, char **)> f, int argc, const char *const *argv, init_params const &params) {
    std::function<int(einsums::program_options::variables_map &)> main_f =
        einsums::util::detail::bind_back(einsums::detail::init_helper, f);
    return detail::init_start_impl(std::move(main_f), argc, argv, params, true);
}

int init(std::function<int()> f, int argc, const char *const *argv, init_params const &params) {
    std::function<int(einsums::program_options::variables_map &)> main_f = std::bind(f);
    return detail::init_start_impl(std::move(main_f), argc, argv, params, true);
}

int init(std::nullptr_t, int argc, const char *const *argv, init_params const &params) {
    std::function<int(einsums::program_options::variables_map &)> main_f;
    return detail::init_start_impl(std::move(main_f), argc, argv, params, true);
}

void start(std::function<int(einsums::program_options::variables_map &)> f, int argc, const char *const *argv, init_params const &params) {
    if (detail::init_start_impl(std::move(f), argc, argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void start(std::function<int(int, char **)> f, int argc, const char *const *argv, init_params const &params) {
    std::function<int(einsums::program_options::variables_map &)> main_f =
        einsums::util::detail::bind_back(einsums::detail::init_helper, f);
    if (detail::init_start_impl(std::move(main_f), argc, argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void start(std::function<int()> f, int argc, const char *const *argv, init_params const &params) {
    std::function<int(einsums::program_options::variables_map &)> main_f = std::bind(f);
    if (detail::init_start_impl(std::move(main_f), argc, argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void start(std::nullptr_t, int argc, const char *const *argv, init_params const &params) {
    std::function<int(einsums::program_options::variables_map &)> main_f;
    if (detail::init_start_impl(std::move(main_f), argc, argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void start(int argc, const char *const *argv, init_params const &params) {
    std::function<int(einsums::program_options::variables_map &)> main_f;
    if (detail::init_start_impl(std::move(main_f), argc, argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void finalize() {
    if (!einsums::detail::is_running()) {
        EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status, "the runtime system is not active (did you already call finalize?)");
    }

    einsums::detail::runtime *rt = einsums::detail::get_runtime_ptr();
    if (nullptr == rt) {
        EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status, "the runtime system is not active (did you already call einsums::stop?)");
    }

    rt->finalize();
}

int stop() {
    if (threads::detail::get_self_ptr()) {
        EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status, "this function cannot be called from a einsums thread");
    }
    // take ownership!
    std::unique_ptr<einsums::detail::runtime> rt(einsums::detail::get_runtime_ptr());
    if (nullptr == rt.get()) {
        EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status, "the runtime system is not active (did you already call einsums::stop?)");
    }

    int result = rt->wait();

    rt->stop();
    rt->rethrow_exception();

    return result;
}

void wait() {
    einsums::detail::runtime *rt = einsums::detail::get_runtime_ptr();
    if (nullptr == rt) {
        EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status, "the runtime system is not active (did you already call einsums::stop?)");
    }

    rt->get_thread_manager().wait();
}

void suspend() {
    if (threads::detail::get_self_ptr()) {
        EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status, "this function cannot be called from a einsums thread");
    }

    einsums::detail::runtime *rt = einsums::detail::get_runtime_ptr();
    if (nullptr == rt) {
        EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status, "the runtime system is not active (did you already call einsums::stop?)");
    }

    rt->suspend();
}

void resume() {
    if (threads::detail::get_self_ptr()) {
        EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status, "this function cannot be called from a einsums thread");
    }

    einsums::detail::runtime *rt = einsums::detail::get_runtime_ptr();
    if (nullptr == rt) {
        EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status, "the runtime system is not active (did you already call einsums::stop?)");
    }

    rt->resume();
}
} // namespace einsums