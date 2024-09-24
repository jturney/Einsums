//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/assert.hpp>
#include <einsums/command_line_handling/command_line_handling.hpp>
#include <einsums/coroutines/detail/context_impl.hpp>
#include <einsums/execution_base/detail/spinlock_deadlock_detection.hpp>
#include <einsums/filesystem.hpp>
#include <einsums/init_runtime/detail/init_logging.hpp>
#include <einsums/init_runtime/init_runtime.hpp>
#include <einsums/lock_registration/detail/register_locks.hpp>
#include <einsums/logging.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/modules/functional.hpp>
#include <einsums/modules/schedulers.hpp>
#include <einsums/modules/timing.hpp>
#include <einsums/program_options/parsers.hpp>
#include <einsums/program_options/variables_map.hpp>
#include <einsums/resource_partitioner/partitioner.hpp>
#include <einsums/runtime/config_entry.hpp>
#include <einsums/runtime/custom_exception_info.hpp>
#include <einsums/runtime/debugging.hpp>
#include <einsums/runtime/report_error.hpp>
#include <einsums/runtime/runtime.hpp>
#include <einsums/runtime/runtime_handlers.hpp>
#include <einsums/runtime/shutdown_function.hpp>
#include <einsums/runtime/startup_function.hpp>
#include <einsums/schedulers/deadlock_detection.hpp>
#include <einsums/string_util/classification.hpp>
#include <einsums/string_util/split.hpp>
#include <einsums/threading/thread.hpp>
#include <einsums/threading_base/detail/get_default_pool.hpp>
#include <einsums/type_support/pack.hpp>
#include <einsums/type_support/unused.hpp>
#include <einsums/util/get_entry_as.hpp>
#include <einsums/version.hpp>

#if defined(EINSUMS_HAVE_MPI)
#    include <einsums/async_mpi/mpi_polling.hpp>
#endif

#if defined(EINSUMS_HAVE_HIP)
#    include <hip/hip_runtime.h>
#    include <whip.hpp>
#endif

#if defined(__bgq__)
#    include <cstdlib>
#endif

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if !defined(EINSUMS_WINDOWS)
#    include <signal.h>
#endif

namespace einsums {
namespace detail {

int init_helper(einsums::program_options::variables_map & /*vm*/, einsums::util::detail::function<int(int, char **)> const &f) {
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
    for (std::size_t i = 0; i != args.size(); ++i) {
        if (0 != args[i].find("--einsums:")) {
            argv[argcount++] = const_cast<char *>(args[i].data());
        } else if (7 == args[i].find("positional", 7)) {
            std::string::size_type p = args[i].find_first_of('=');
            if (p != std::string::npos) {
                args[i]          = args[i].substr(p + 1);
                argv[argcount++] = const_cast<char *>(args[i].data());
            }
        }
    }

    // add a single nullptr in the end as some application rely on that
    argv[argcount] = nullptr;

    // Invoke custom startup functions
    return f(static_cast<int>(argcount), argv.data());
}

void activate_global_options(detail::command_line_handling &cmdline) {
#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
    einsums::threads::coroutines::detail::posix::use_guard_pages = cmdline._rtcfg.use_stack_guard_pages();
#endif
#ifdef EINSUMS_HAVE_VERIFY_LOCKS
    if (cmdline._rtcfg.enable_lock_detection()) {
        einsums::util::enable_lock_detection();
        einsums::util::trace_depth_lock_detection(cmdline._rtcfg.trace_depth());
    } else {
        einsums::util::disable_lock_detection();
    }
#endif
#ifdef EINSUMS_HAVE_THREAD_DEADLOCK_DETECTION
    einsums::threads::detail::set_deadlock_detection_enabled(cmdline._rtcfg.enable_deadlock_detection());
#endif
#ifdef EINSUMS_HAVE_SPINLOCK_DEADLOCK_DETECTION
    einsums::util::detail::set_spinlock_break_on_deadlock_enabled(cmdline._rtcfg.enable_spinlock_deadlock_detection());
    einsums::util::detail::set_spinlock_deadlock_detection_limit(cmdline._rtcfg.get_spinlock_deadlock_detection_limit());
    einsums::util::detail::set_spinlock_deadlock_warning_limit(cmdline._rtcfg.get_spinlock_deadlock_warning_limit());
#endif
#ifdef EINSUMS_HAVE_MPI
    einsums::mpi::experimental::set_completion_mode(
        einsums::detail::get_entry_as<std::size_t>(cmdline._rtcfg, "einsums.mpi.completion_mode", 0));
#endif
    init_logging(cmdline._rtcfg);
}

struct dump_config {
    explicit dump_config(einsums::detail::runtime const &rt) : rt_(std::cref(rt)) {}

    void operator()() const {
        std::cout << "Configuration after runtime start:\n";
        std::cout << "----------------------------------\n";
        rt_.get().get_config().dump(0, std::cout);
        std::cout << "----------------------------------\n";
    }

    std::reference_wrapper<einsums::detail::runtime const> rt_;
};

///////////////////////////////////////////////////////////////////////
void add_startup_functions(einsums::detail::runtime &rt, einsums::program_options::variables_map &vm, startup_function_type startup,
                           shutdown_function_type shutdown) {
    if (vm.count("einsums:app-config")) {
        std::string config(vm["einsums:app-config"].as<std::string>());
        rt.get_config().load_application_configuration(config.c_str());
    }

    if (!!startup)
        rt.add_startup_function(std::move(startup));

    if (!!shutdown)
        rt.add_shutdown_function(std::move(shutdown));

    if (vm.count("einsums:dump-config-initial")) {
        std::cout << "Configuration after runtime construction:\n";
        std::cout << "-----------------------------------------\n";
        rt.get_config().dump(0, std::cout);
        std::cout << "-----------------------------------------\n";
    }

    if (vm.count("einsums:dump-config"))
        rt.add_startup_function(dump_config(rt));
}

///////////////////////////////////////////////////////////////////////
int run(einsums::detail::runtime &rt, einsums::util::detail::function<int(einsums::program_options::variables_map &vm)> const &f,
        einsums::program_options::variables_map &vm, startup_function_type startup, shutdown_function_type shutdown) {
    add_startup_functions(rt, vm, std::move(startup), std::move(shutdown));

    // Run this runtime instance using the given function f.
    if (!f.empty())
        return rt.run(einsums::util::detail::bind_front(f, vm));

    // Run this runtime instance without a einsums_main
    return rt.run();
}

int start(einsums::detail::runtime &rt, einsums::util::detail::function<int(einsums::program_options::variables_map &vm)> const &f,
          einsums::program_options::variables_map &vm, startup_function_type startup, shutdown_function_type shutdown) {
    add_startup_functions(rt, vm, std::move(startup), std::move(shutdown));

    if (!f.empty()) {
        // Run this runtime instance using the given function f.
        return rt.start(einsums::util::detail::bind_front(f, vm));
    }

    // Run this runtime instance without a einsums_main
    return rt.start();
}

int run_or_start(bool blocking, std::unique_ptr<einsums::detail::runtime> rt, detail::command_line_handling &cfg,
                 startup_function_type startup, shutdown_function_type shutdown) {
    if (blocking) {
        return run(*rt, cfg._einsums_main_f, cfg._vm, std::move(startup), std::move(shutdown));
    }

    // non-blocking version
    start(*rt, cfg._einsums_main_f, cfg._vm, std::move(startup), std::move(shutdown));

    // pointer to runtime is stored in TLS
    [[maybe_unused]] einsums::detail::runtime *p = rt.release();

    return 0;
}

////////////////////////////////////////////////////////////////////////
void init_environment(detail::command_line_handling &cmdline) {
    EINSUMS_UNUSED(einsums::detail::filesystem::initial_path());

    einsums::detail::set_assertion_handler(&einsums::detail::assertion_handler);
    einsums::detail::set_custom_exception_info_handler(&einsums::detail::custom_exception_info);
    einsums::detail::set_pre_exception_handler(&einsums::detail::pre_exception_handler);
    einsums::set_thread_termination_handler([](std::exception_ptr const &e) { report_error(e); });
    einsums::detail::set_get_full_build_string(&einsums::full_build_string);
#if defined(EINSUMS_HAVE_VERIFY_LOCKS)
    einsums::util::set_registered_locks_error_handler(&einsums::detail::registered_locks_error_handler);
    einsums::util::set_register_locks_predicate(&einsums::detail::register_locks_predicate);
#endif
    if (einsums::detail::get_entry_as<bool>(cmdline._rtcfg, "einsums.install_signal_handlers", false)) {
        set_signal_handlers();
    }

    einsums::threads::detail::set_get_default_pool(&einsums::detail::get_default_pool);

#if defined(__bgq__) || defined(__bgqion__)
    unsetenv("LANG");
    unsetenv("LC_CTYPE");
    unsetenv("LC_NUMERIC");
    unsetenv("LC_TIME");
    unsetenv("LC_COLLATE");
    unsetenv("LC_MONETARY");
    unsetenv("LC_MESSAGES");
    unsetenv("LC_PAPER");
    unsetenv("LC_NAME");
    unsetenv("LC_ADDRESS");
    unsetenv("LC_TELEPHONE");
    unsetenv("LC_MEASUREMENT");
    unsetenv("LC_IDENTIFICATION");
    unsetenv("LC_ALL");
#endif
}

///////////////////////////////////////////////////////////////////////
int run_or_start(einsums::util::detail::function<int(einsums::program_options::variables_map &vm)> const &f, int argc,
                 const char *const *argv, init_params const &params, bool blocking) {
    if (einsums::detail::get_runtime_ptr() != nullptr) {
        EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status, "einsums::start/init", "runtime already initialized");
    }

    einsums::detail::command_line_handling cmdline{einsums::util::runtime_configuration(argv[0]), params.cfg, f};

    auto cmdline_result = cmdline.call(params.desc_cmdline, argc, argv);

    einsums::detail::affinity_data affinity_data{};
    affinity_data.init(einsums::detail::get_entry_as<std::size_t>(cmdline._rtcfg, "einsums.os_threads", 0),
                       einsums::detail::get_entry_as<std::size_t>(cmdline._rtcfg, "einsums.cores", 0),
                       einsums::detail::get_entry_as<std::size_t>(cmdline._rtcfg, "einsums.pu_offset", 0),
                       einsums::detail::get_entry_as<std::size_t>(cmdline._rtcfg, "einsums.pu_step", 0), 0,
                       cmdline._rtcfg.get_entry("einsums.affinity", ""), cmdline._rtcfg.get_entry("einsums.bind", ""),
                       !einsums::detail::get_entry_as<bool>(cmdline._rtcfg, "einsums.ignore_process_mask", false));

    einsums::resource::partitioner rp = einsums::resource::detail::make_partitioner(params.rp_mode, cmdline._rtcfg, affinity_data);

    activate_global_options(cmdline);

    init_environment(cmdline);

    switch (cmdline_result) {
    case command_line_handling_result::success:
        break;
    case command_line_handling_result::exit:
        return 0;
    }

    // If thread_pools initialization in user main
    if (params.rp_callback) {
        params.rp_callback(rp, cmdline._vm);
    }

    // Setup all internal parameters of the resource_partitioner
    rp.configure_pools();

#if defined(EINSUMS_HAVE_HIP)
    EINSUMS_LOG(info, "run_local: initialize HIP");
    whip::check_error(hipInit(0));
#endif

    // Initialize and start the einsums runtime.
    EINSUMS_LOG(info, "run_local: create runtime");

    // Build and configure this runtime instance.
    std::unique_ptr<einsums::detail::runtime> rt;

    // Command line handling should have updated this by now.
    EINSUMS_LOG(info, "creating local runtime");
    rt.reset(new einsums::detail::runtime(cmdline._rtcfg, true));

    return run_or_start(blocking, std::move(rt), cmdline, std::move(params.startup), std::move(params.shutdown));
}

int init_start_impl(einsums::util::detail::function<int(einsums::program_options::variables_map &)> f, int argc, const char *const *argv,
                    init_params const &params, bool blocking) {
    if (argc == 0 || argv == nullptr) {
        argc = dummy_argc;
        argv = dummy_argv;
    }

#if defined(__FreeBSD__)
    freebsd_environ = environ;
#endif
    // set a handler for std::abort
    [[maybe_unused]] auto signal_handler = std::signal(SIGABRT, einsums::detail::on_abort);
    [[maybe_unused]] auto exit_result    = std::atexit(einsums::detail::on_exit);
#if defined(EINSUMS_HAVE_CXX11_STD_QUICK_EXIT)
    [[maybe_unused]] auto quick_exit_result = std::at_quick_exit(einsums::detail::on_exit);
#endif
    return run_or_start(f, argc, argv, params, blocking);
}
} // namespace detail

int init(std::function<int(einsums::program_options::variables_map &)> f, int argc, const char *const *argv, init_params const &params) {
    return detail::init_start_impl(std::move(f), argc, argv, params, true);
}

int init(std::function<int(int, char **)> f, int argc, const char *const *argv, init_params const &params) {
    einsums::util::detail::function<int(einsums::program_options::variables_map &)> main_f =
        einsums::util::detail::bind_back(einsums::detail::init_helper, f);
    return detail::init_start_impl(std::move(main_f), argc, argv, params, true);
}

int init(std::function<int()> f, int argc, const char *const *argv, init_params const &params) {
    einsums::util::detail::function<int(einsums::program_options::variables_map &)> main_f = einsums::util::detail::bind(f);
    return detail::init_start_impl(std::move(main_f), argc, argv, params, true);
}

int init(std::nullptr_t, int argc, const char *const *argv, init_params const &params) {
    einsums::util::detail::function<int(einsums::program_options::variables_map &)> main_f;
    return detail::init_start_impl(std::move(main_f), argc, argv, params, true);
}

void start(std::function<int(einsums::program_options::variables_map &)> f, int argc, const char *const *argv, init_params const &params) {
    if (detail::init_start_impl(std::move(f), argc, argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void start(std::function<int(int, char **)> f, int argc, const char *const *argv, init_params const &params) {
    einsums::util::detail::function<int(einsums::program_options::variables_map &)> main_f =
        einsums::util::detail::bind_back(einsums::detail::init_helper, f);
    if (detail::init_start_impl(std::move(main_f), argc, argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void start(std::function<int()> f, int argc, const char *const *argv, init_params const &params) {
    einsums::util::detail::function<int(einsums::program_options::variables_map &)> main_f = einsums::util::detail::bind(f);
    if (detail::init_start_impl(std::move(main_f), argc, argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void start(std::nullptr_t, int argc, const char *const *argv, init_params const &params) {
    einsums::util::detail::function<int(einsums::program_options::variables_map &)> main_f;
    if (detail::init_start_impl(std::move(main_f), argc, argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void start(int argc, const char *const *argv, init_params const &params) {
    einsums::util::detail::function<int(einsums::program_options::variables_map &)> main_f;
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
