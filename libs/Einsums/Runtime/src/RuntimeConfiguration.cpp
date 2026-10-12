//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Hardware/Profile.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Options.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime/RuntimeConfiguration.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>
#include <Einsums/Version.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
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

/// Which options a hardware profile may set: performance options a tune measures, and never one
/// that decides where the profile is or which machine it describes.
std::string profile_option_check(std::string_view name) {
    static constexpr std::array<std::string_view, 3> locating{"einsums:hardware:profile", "einsums:hardware:site-profiles",
                                                              "einsums:cache-dir"};
    static constexpr std::array<std::string_view, 5> performance{"ComputeGraph Passes", "PackedGemm", "Hardware", "Buffer Allocator",
                                                                 "HPTT"};
    if (std::ranges::find(locating, name) != locating.end()) {
        return "it locates the profile, so a profile cannot set it";
    }
    for (auto const &option : cl::registered_options()) {
        if (option.name == name) {
            if (std::ranges::find(performance, option.category) != performance.end()) {
                return {};
            }
            return fmt::format("it is a {} option, and a profile sets only performance options", option.category);
        }
    }
    return "no such option";
}

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
        EINSUMS_THROW_EXCEPTION(SystemError, "unable to find executable filename");
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
        EINSUMS_THROW_EXCEPTION(SystemError, "unable to find executable filename");
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
    // Will eventually read a template of default settings.
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
    // pre_initialize (run by the constructor) has set the defaults.

    // Options read the environment too (einsums:log:level reads EINSUMS_LOG_LEVEL); precedence is
    // default < config file < environment < command line.
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

    hardware::set_profile_option_check(&profile_option_check);

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

void RuntimeConfiguration::apply_hardware_profile() {
    auto const options = hardware::profile_options();
    if (options.empty()) {
        return;
    }
    std::vector<std::string> unknown;
    auto const               applied = cl::parse_internal(original, "Einsums", full_version_as_string(), &options, &unknown);
    if (applied.ok) {
        _unknown_arguments = std::move(unknown);
        EINSUMS_LOG_INFO("hardware profile: applied {} option(s) from {}", options.size(), hardware::profile_path());
        return;
    }
    // The parser has said which value it rejected. Parse again without the profile, so every
    // option is back to what the environment and the command line made it.
    EINSUMS_LOG_WARN("hardware profile: the options in {} were not applied, since the parser rejected one", hardware::profile_path());
    _unknown_arguments.clear();
    cl::parse_internal(original, "Einsums", full_version_as_string(), nullptr, &_unknown_arguments);
}

EINSUMS_NAMESPACE_END()
