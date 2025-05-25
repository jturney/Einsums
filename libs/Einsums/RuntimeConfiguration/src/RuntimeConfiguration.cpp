//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#if defined(EINSUMS_WINDOWS)
#    include <process.h>
#elif defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#if defined(EINSUMS_WINDOWS)
#    include <windows.h>
#elif defined(__linux) || defined(linux) || defined(__linux__)
#    include <filesystem>
#elif __APPLE__
#    include <mach-o/dyld.h>
#endif

#include <argparse/argparse.hpp>

namespace einsums {
namespace detail {

struct EINSUMS_EXPORT ArgumentList final : design_pats::Lockable<std::mutex> {
    EINSUMS_SINGLETON_DEF(ArgumentList)

  public:
    std::list<std::function<void(argparse::ArgumentParser &)>> argument_functions{};

  private:
    explicit ArgumentList() = default;
};

EINSUMS_SINGLETON_IMPL(ArgumentList)

std::string get_executable_filename() {
    std::string r;

#if defined(EINSUMS_WINDOWS)
    char exe_path[MAX_PATH + 1] = {'\0'};
    if (!GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path))) {
        EINSUMS_THROW_EXCEPTION(system_error, "unable to find executable filename");
    }
    r = exe_path;
#elif defined(__linux) || defined(linux) || defined(__linux__)
    r     = std::filesystem::canonical("/proc/self/exe").string();
    errno = 0; // errno seems to be set by the previous call. However, since the call does not throw an exception,
    // and the specification for the call requires it to throw an exception when it encounters an error,
    // it can be assumed that the error is actually not important.
#elif defined(__APPLE__)
    char          exe_path[PATH_MAX + 1];
    std::uint32_t len = sizeof(exe_path) / sizeof(exe_path[0]);

    if (0 != _NSGetExecutablePath(exe_path, &len)) {
        EINSUMS_THROW_EXCEPTION(system_error, "unable to find executable filename");
    }
    exe_path[len - 1] = '\0';
    r                 = exe_path;
#else
#    error Unsupported platform
#endif

    return r;
}

std::string get_executable_prefix() {
    std::filesystem::path p(get_executable_filename());
    std::string           prefix = p.parent_path().parent_path().string();

    return prefix;
}
} // namespace detail

void register_arguments(std::function<void(argparse::ArgumentParser &)> func) {
    auto &argument_list = detail::ArgumentList::get_singleton();
    auto  lock          = std::lock_guard(argument_list);

    argument_list.argument_functions.push_back(func);
}

void RuntimeConfiguration::pre_initialize() {
    /*
     * This routine will eventually contain a "master" yaml template that
     * will include all the default settings for Einsums and its subsystems.
     *
     * Once a yaml or some other file settings format is decided on and brought
     * in that file will be used after initially using this one.
     */
    std::vector<std::string> lines = {
        // clang-format off
        "system:",
        "    pid: " + std::to_string(getpid()),
        "    executable_prefix: " + detail::get_executable_prefix(),
        "einsums:",
        "    master_yaml_file: ${system.executable_prefix}"
        "    "
        // clang-format on
    };

    /*
     * Acquire locks for the different maps.
     */
    auto            &global_config  = GlobalConfigMap::get_singleton();
    auto            &global_strings = global_config.get_string_map()->get_value();
    auto            &global_ints    = global_config.get_int_map()->get_value();
    auto            &global_doubles = global_config.get_double_map()->get_value();
    auto            &global_bools   = global_config.get_bool_map()->get_value();
    std::scoped_lock lock{*global_config.get_string_map(), *global_config.get_int_map(), *global_config.get_double_map(),
                          *global_config.get_bool_map()};

    // For now set the values to their default values.
    global_strings["executable-prefix"] = detail::get_executable_prefix();
    global_ints["pid"]                  = getpid();
}

RuntimeConfiguration::RuntimeConfiguration(int argc, char const *const argv[],
                                           std::function<void(argparse::ArgumentParser &)> const &user_command_line)
    : original(argc) {

    // Make a copy. If a new argv was derived from the argv on entry, then it may not
    // be available at every point. Also, making it a vector makes it easier to use.
    for (int i = 0; i < argc; i++) {
        original[i] = std::string(argv[i]);
    }

    pre_initialize();

    parse_command_line(user_command_line);
}

RuntimeConfiguration::RuntimeConfiguration(std::vector<std::string> const                        &argv,
                                           std::function<void(argparse::ArgumentParser &)> const &user_command_line)
    : original(argv) {
    pre_initialize();

    parse_command_line(user_command_line);
}

std::vector<std::string>
RuntimeConfiguration::parse_command_line(std::function<void(argparse::ArgumentParser &)> const &user_command_line) {
    // Imperative that pre_initialize is called first as it is responsible for setting
    // default values. This is done in the constructor.
    // There should be a mechanism that allows the user to change the program name.
    if (!original[0].empty()) {
        argument_parser = std::make_unique<argparse::ArgumentParser>(original[0]);
    } else {
        argument_parser = std::make_unique<argparse::ArgumentParser>("einsums");
    }

    /*
     * Acquire locks for the different maps.
     */
    auto &global_config  = GlobalConfigMap::get_singleton();
    auto &global_strings = global_config.get_string_map()->get_value();
    auto &global_ints    = global_config.get_int_map()->get_value();
    auto &global_doubles = global_config.get_double_map()->get_value();
    auto &global_bools   = global_config.get_bool_map()->get_value();
    {
        std::scoped_lock lock{*global_config.get_string_map(), *global_config.get_int_map(), *global_config.get_double_map(),
                              *global_config.get_bool_map()};

        argument_parser->add_argument("--einsums:no-install-signal-handlers")
            .default_value(true)
            .implicit_value(false)
            .help("do not install signal handlers")
            .store_into(global_bools["install-signal-handlers"]);

        argument_parser->add_argument("--einsums:no-attach-debugger")
            .default_value(true)
            .implicit_value(false)
            .help("do not provide mechanism to attach debugger on detected errors")
            .store_into(global_bools["attach-debugger"]);

        argument_parser->add_argument("--einsums:no-diagnostics-on-terminate")
            .default_value(true)
            .implicit_value(false)
            .help("do not print additional diagnostic information on termination")
            .store_into(global_bools["diagnostics-on-terminate"]);

        argument_parser
            ->add_argument("--einsums:log-level")
#ifdef EINSUMS_DEBUG
            .default_value<std::int64_t>(SPDLOG_LEVEL_DEBUG)
#else
            .default_value<std::int64_t>(SPDLOG_LEVEL_INFO)
#endif
            .help("set log level")
            .choices(0, 1, 2, 3, 4)
            .store_into(global_ints["log-level"]);
        argument_parser->add_argument("--einsums:log-destination")
            .default_value("cerr")
            .help("set log destination")
            .choices("cerr", "cout")
            .store_into(global_strings["log-destination"]);
        argument_parser->add_argument("--einsums:log-format")
            .default_value("[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%-8l%$] [%s:%#/%!] %v")
            .store_into(global_strings["log-format"]);

        argument_parser->add_argument("--einsums:no-profiler-report")
            .default_value(true)
            .implicit_value(false)
            .help("generate profiling report")
            .store_into(global_bools["profiler-report"]);

        argument_parser->add_argument("--einsums:profiler-filename")
            .default_value("profile.txt")
            .help("filename of the profiling report")
            .store_into(global_strings["profiler-filename"]);
        argument_parser->add_argument("--einsums:no-profiler-append")
            .default_value(true)
            .implicit_value(false)
            .help("append to an existing file")
            .store_into(global_bools["profiler-append"]);
    }

    {
        auto &argument_list = detail::ArgumentList::get_singleton();

        auto lock = std::lock_guard(argument_list);

        // Inject module-specific command lines.
        for (auto &func : argument_list.argument_functions) {
            func(*argument_parser);
        }
    }

    // Allow the user to inject their own command line options
    if (user_command_line) {
        user_command_line(*argument_parser);
    }

    try {
        global_config.lock();
        auto out = argument_parser->parse_known_args(original);
        global_config.unlock();
        return out;
    } catch (std::exception const &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << argument_parser;
        std::exit(1);
    }
}

} // namespace einsums