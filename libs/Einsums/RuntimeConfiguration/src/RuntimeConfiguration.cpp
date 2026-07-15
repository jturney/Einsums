//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/CommandLine.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
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

namespace einsums {

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

std::string get_executable_prefix() {
    std::filesystem::path const p(get_executable_filename());
    std::string                 prefix = p.parent_path().parent_path().string();

    return prefix;
}
} // namespace

void register_arguments(std::function<void()> const &func) {
    auto                  &argument_list = ArgumentList::get_singleton();
    std::scoped_lock const lock(argument_list);

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
    std::vector<std::string> const lines = {
        // clang-format off
        "system:",
        "    pid: " + std::to_string(getpid()),
        "    executable_prefix: " + get_executable_prefix(),
        "einsums:",
        "    master_yaml_file: ${system.executable_prefix}",
        "    "
        // clang-format on
    };

    /*
     * Acquire locks for the different maps.
     *
     * Use std::lock_guard on the GlobalConfigMap itself rather than
     * std::scoped_lock over the four sub-maps. GlobalConfigMap::lock() takes
     * the sub-maps in a fixed order (str -> int -> double -> bool), which
     * matches every other call site that holds the four maps simultaneously
     * (e.g. add_Einsums_Tensor_arguments in Tensor/src/InitModule.cpp).
     * std::scoped_lock would dispatch to std::lock's deadlock-avoidance
     * algorithm, which acquires in a *dynamic* order; TSan then sees two
     * incompatible acquisition orders for the same four mutexes and reports
     * a (real but harmless under std::lock) lock-order inversion. Going
     * through the canonical fixed order keeps all call sites consistent.
     */
    auto                  &global_config  = GlobalConfigMap::get_singleton();
    auto                  &global_strings = global_config.get_string_map()->get_value();
    auto                  &global_ints    = global_config.get_int_map()->get_value();
    auto                  &global_doubles = global_config.get_double_map()->get_value();
    auto                  &global_bools   = global_config.get_bool_map()->get_value();
    std::scoped_lock const lock(global_config);

    // For now set the values to their default values.
    global_strings["executable-prefix"] = get_executable_prefix();
    global_ints["pid"]                  = getpid();

    // Silence unused-variable warnings for the references we only kept around
    // for symmetry with the rest of the module.
    (void)global_doubles;
    (void)global_bools;
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

    /*
     * Acquire locks for the different maps.
     */
    auto &global_config  = GlobalConfigMap::get_singleton();
    auto &global_strings = global_config.get_string_map()->get_value();
    auto &global_ints    = global_config.get_int_map()->get_value();
    auto &global_doubles = global_config.get_double_map()->get_value();
    auto &global_bools   = global_config.get_bool_map()->get_value();
    {
        // See pre_initialize for the rationale: use the GlobalConfigMap's
        // own fixed-order lock(), not std::scoped_lock over the four
        // sub-maps (which would dispatch to std::lock with dynamic
        // ordering and create a TSan lock-order cycle with every other
        // call site that locks the maps via GlobalConfigMap::lock()).
        std::scoped_lock const lock(global_config);

        // These options are static but all use Location to initialize the
        // members of the parent class.
        static cl::OptionCategory debugCategory("Debug");
        static cl::Flag const     noInstallSignalHandlers("einsums:debug:no-install-signal-handlers", {}, "Do not install signal handlers",
                                                          debugCategory, cl::Location(global_bools["install-signal-handlers"]),
                                                          cl::Default(true), cl::ImplicitValue(false));

        static cl::Flag const noAttachDebugger("einsums:debug:no-attach-debugger", {},
                                               "Do not provide a mechanism to attach a debugger on detected errors", debugCategory,
                                               cl::Location(global_bools["attach-debugger"]), cl::Default(true), cl::ImplicitValue(false));

        static cl::Flag const noDiagnosticsOnTerminate(
            "einsums:debug:no-diagnostics-on-terminate", {}, "Print additional diagnostic information on termination", debugCategory,
            cl::Location(global_bools["diagnostics-on-terminate"]), cl::Default(true), cl::ImplicitValue(false));

        static cl::OptionCategory     logCategory("Logging");
        static cl::Opt<int64_t> const logLevel("einsums:log:level", {}, "Log level", logCategory, cl::Location(global_ints["log-level"]),
                                               cl::Default(static_cast<int64_t>(
#if defined(EINSUMS_DEBUG)
                                                   SPDLOG_LEVEL_DEBUG
#else
                                                   SPDLOG_LEVEL_ERROR
#endif
                                                   )),
                                               cl::RangeBetween(0, 4), cl::ValueName("LogLevel"));

        static cl::Opt<std::string> const logDestination("einsums:log:destination", {}, "Log destination", logCategory,
                                                         cl::Location(global_strings["log-destination"]), cl::Default(std::string("cerr")));

        static cl::Opt<std::string> const logFormat("einsums:log:format", {}, "Log format", logCategory,
                                                    cl::Location(global_strings["log-format"]),
                                                    cl::Default(std::string("[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%-8l%$] [%s:%#/%!] %v")));

        static cl::OptionCategory profileCategory("Profile");
        static cl::Flag const     noProfileReport("einsums:profile:no-report", {}, "Don't generate profile report", profileCategory,
                                                  cl::Location(global_bools["profiler-report"]), cl::Default(true), cl::ImplicitValue(false));

        static cl::Opt<std::string> const profileFilename("einsums:profile:filename", {}, "Generate profile filename", profileCategory,
                                                          cl::Location(global_strings["profiler-filename"]),
                                                          cl::Default(std::string("profile.txt")), cl::ValueName("filename"));

        static cl::Opt<bool> const noProfileAppend("einsums:profile:no-append", {}, "Don't append to profile file", profileCategory,
                                                   cl::Location(global_bools["profiler-append"]), cl::Default(true),
                                                   cl::ImplicitValue(false), cl::ValueName("N"));

        static cl::Flag const profileDetailed("einsums:profile:detailed", {}, "Print detailed profile report", profileCategory,
                                              cl::Location(global_bools["profiler-detailed"]), cl::Default(false), cl::ImplicitValue(true));

        static cl::Opt<std::string> const profileSave("einsums:profile:save", {}, "Save profile session JSON for imgui viewer",
                                                      profileCategory, cl::Location(global_strings["profiler-save"]),
                                                      cl::Default(std::string("")), cl::ValueName("filename"));

        static cl::Opt<int64_t> const profilePort("einsums:profile:port", {}, "Profile server port", profileCategory,
                                                  cl::Location(global_ints["profiler-port"]), cl::Default(static_cast<int64_t>(19216)),
                                                  cl::ValueName("PORT"));

        static cl::Flag const waitForViewer("einsums:profile:wait-for-viewer", {}, "Wait for profiler viewer to connect before running",
                                            profileCategory, cl::Location(global_bools["profiler-wait-for-viewer"]), cl::Default(false),
                                            cl::ImplicitValue(true));

        static cl::OptionCategory         hpttCategory("HPTT");
        static cl::Opt<std::string> const hpttSelectionMethod(
            "einsums:hptt:selection-method", {}, "HPTT plan selection method (estimate, measure, patient, crazy)", hpttCategory,
            cl::Location(global_strings["hptt-selection-method"]), cl::Default(std::string("estimate")), cl::ValueName("METHOD"));

        static cl::OptionCategory         passCategory("ComputeGraph Passes");
        static cl::Opt<std::string> const passDisable(
            "einsums:pass:disable", {}, "Comma-separated list of optimization pass names to skip (e.g. CSE,Reorder)", passCategory,
            cl::Location(global_strings["pass-disable"]), cl::Default(std::string("")), cl::ValueName("PASSES"));
        static cl::Flag const passAnalyze("einsums:pass:analyze", {},
                                          "Run all passes in analysis-only mode (report findings but don't modify the graph)", passCategory,
                                          cl::Location(global_bools["pass-analyze"]), cl::Default(false), cl::ImplicitValue(true));
        static cl::Flag const passVerbose("einsums:pass:verbose", {}, "Log node count and timing before/after each optimization pass",
                                          passCategory, cl::Location(global_bools["pass-verbose"]), cl::Default(false),
                                          cl::ImplicitValue(true));
    }

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

    try {
        auto pr = cl::parse(original, "Einsums", full_version_as_string(), &_unknown_arguments);

        if (!pr.ok) {
            std::exit(pr.exit_code);
        }
    } catch (std::exception const &) {
        std::exit(1);
    }
}

} // namespace einsums