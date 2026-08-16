//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Options.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>
#include <Einsums/Version.hpp>

#include <filesystem>
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

EINSUMS_NAMESPACE_BEGIN()

namespace {

// NOT EINSUMS_EXPORT: dllexport requires external linkage, and this type
// lives in an anonymous namespace on purpose (TU-local singleton).
struct ArgumentList final : design_pats::Lockable<std::mutex> {
    EINSUMS_SINGLETON_DEF(ArgumentList)

    std::list<std::function<void()>> argument_functions{};

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
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
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

} // namespace

std::string executable_prefix() {
    std::filesystem::path const p(get_executable_filename());
    return p.parent_path().parent_path().string();
}

void register_arguments(std::function<void()> const &func) {
    auto                  &argument_list = ArgumentList::get_singleton();
    std::scoped_lock const lock(argument_list);

    argument_list.argument_functions.push_back(func);
}

void RuntimeConfiguration::pre_initialize() {
    /*
     * This routine will eventually read a "master" yaml template carrying the
     * default settings for Einsums and its subsystems.
     *
     * The process id and the install prefix used to be written into the config
     * maps here. They are facts about the running process, not options anyone
     * sets, so they are current_process_id() and executable_prefix() instead;
     * nothing read them through the map.
     */
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
RuntimeConfiguration::RuntimeConfiguration(int argc, char const *const argv[], std::function<void()> const &user_command_line)
    : original(argc) {

    // Make a copy. If a new argv was derived from the argv on entry, then it may not
    // be available at every point. Also, making it a vector makes it easier to use.
    for (int i = 0; i < argc; i++) {
        original[i] = std::string(argv[i]);
    }

    pre_initialize();

    parse_command_line(user_command_line);
}

RuntimeConfiguration::RuntimeConfiguration(std::vector<std::string> const &argv, std::function<void()> const &user_command_line)
    : original(argv) {
    pre_initialize();

    parse_command_line(user_command_line);
}

void RuntimeConfiguration::parse_command_line(std::function<void()> const &user_command_line) {
    // Imperative that pre_initialize is called first as it is responsible for setting
    // default values. This is done in the constructor.
    // There should be a mechanism that allows the user to change the program name.

    // Every option is readable from the environment under a name derived from
    // its own: einsums:log:level reads EINSUMS_LOG_LEVEL. Precedence runs
    // default < config file < environment < command line, so an inherited
    // variable configures a job without editing its command line, and the
    // command line still wins when it says so.
    cl::Registry::instance().set_env_prefix("EINSUMS");

    {
        auto &argument_list = ArgumentList::get_singleton();

        std::scoped_lock const lock(argument_list);

        // Inject module-specific command lines.
        for (auto &func : argument_list.argument_functions) {
            func();
        }
    }

    // Allow the user to inject their own command line options
    if (user_command_line) {
        user_command_line();
    }

    // Debug builds catch two descriptors resolving to one key here, before a
    // silent read of the wrong slot has a chance to look like a default.
    cl::verify_registered_options();

    try {
        auto pr = cl::parse(original, "Einsums", full_version_as_string(), &_unknown_arguments);

        if (!pr.ok) {
            std::exit(pr.exit_code);
        }
    } catch (std::exception const &) {
        std::exit(1);
    }

    // Registration belongs to this window and nowhere else. Values stay
    // writable: a slot is atomic, and writing one was never the problem.
    cl::freeze_registry();
}

EINSUMS_NAMESPACE_END()