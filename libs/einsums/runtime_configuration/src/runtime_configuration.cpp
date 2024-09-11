//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/modules/errors.hpp>
#include <einsums/runtime_configuration/init_ini_data.hpp>
#include <einsums/runtime_configuration/runtime_configuration.hpp>
#include <einsums/string_util/classification.hpp>
#include <einsums/string_util/split.hpp>
#include <einsums/type_support/unused.hpp>

#include <filesystem>

#if defined(EINSUMS_WINDOWS)
#    include <process.h>
#elif defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#include <limits>

#if defined(EINSUMS_WINDOWS)
#    include <windows.h>
#elif defined(__linux) || defined(linux) || defined(__linux__)
#    include <linux/limits.h>
#    include <sys/stat.h>
#    include <unistd.h>
#    include <vector>
#elif __APPLE__
#    include <mach-o/dyld.h>
#elif defined(__FreeBSD__)
#    include <algorithm>
#    include <iterator>
#    include <sys/sysctl.h>
#    include <sys/types.h>
#    include <vector>
#endif

namespace einsums::detail {

std::string get_executable_filename(char const *argv0) {
    std::string r;

#if defined(EINSUMS_WINDOWS)
    EINSUMS_UNUSED(argv0);

    char exe_path[MAX_PATH + 1] = {'\0'};
    if (!GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path))) {
        EINSUMS_THROW_EXCEPTION(einsums::error::dynamic_link_failure, "get_executable_filename", "unable to find executable filename");
    }
    r = exe_path;

#elif defined(__linux) || defined(linux) || defined(__linux__)
    char    buf[PATH_MAX + 1];
    ssize_t length = ::readlink("/proc/self/exe", buf, sizeof(buf));

    if (length != -1) {
        buf[length] = '\0';
        r           = buf;
        return r;
    }

    std::string argv0_(argv0);

    // REVIEW: Should we resolve symlinks at any point here?
    if (argv0_.length() > 0) {
        // Check for an absolute path.
        if (argv0_[0] == '/')
            return argv0_;

        // Check for a relative path.
        if (argv0_.find('/') != std::string::npos) {
            // Get the current working directory.

            // NOTE: getcwd does give you a null terminated string,
            // while readlink (above) does not.
            if (::getcwd(buf, PATH_MAX)) {
                r = buf;
                r += '/';
                r += argv0_;
                return r;
            }
        }

        // Search PATH
        char const *epath = ::getenv("PATH");
        if (epath) {
            std::vector<std::string> path_dirs;

            einsums::string_util::split(path_dirs, epath, einsums::string_util::is_any_of(":"),
                                        einsums::string_util::token_compress_mode::on);

            for (std::uint64_t i = 0; i < path_dirs.size(); ++i) {
                r = path_dirs[i];
                r += '/';
                r += argv0_;

                // Can't use Boost.Filesystem as it doesn't let me access
                // st_uid and st_gid.
                struct stat s;

                // Make sure the file is executable and shares our
                // effective uid and gid.
                // NOTE: If someone was using a einsums application that was
                // seteuid'd to root, this may fail.
                if (0 == ::stat(r.c_str(), &s))
                    if ((s.st_uid == ::geteuid()) && (s.st_mode & S_IXUSR) && (s.st_gid == ::getegid()) && (s.st_mode & S_IXGRP) &&
                        (s.st_mode & S_IXOTH))
                        return r;
            }
        }
    }

    EINSUMS_THROW_EXCEPTION(einsums::error::dynamic_link_failure, "unable to find executable filename");

#elif defined(__APPLE__)
    EINSUMS_UNUSED(argv0);

    char          exe_path[PATH_MAX + 1];
    std::uint32_t len = sizeof(exe_path) / sizeof(exe_path[0]);

    if (0 != _NSGetExecutablePath(exe_path, &len)) {
        EINSUMS_THROW_EXCEPTION(einsums::error::dynamic_link_failure, "unable to find executable filename");
    }
    exe_path[len - 1] = '\0';
    r                 = exe_path;

#elif defined(__FreeBSD__)
    EINSUMS_UNUSED(argv0);

    int    mib[] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
    size_t cb    = 0;
    sysctl(mib, 4, nullptr, &cb, nullptr, 0);
    if (cb) {
        std::vector<char> buf(cb);
        sysctl(mib, 4, &buf[0], &cb, nullptr, 0);
        std::copy(buf.begin(), buf.end(), std::back_inserter(r));
    }

#else
#    error Unsupported platform
#endif

    return r;
}

std::string get_executable_prefix(char const *argv0 = nullptr) {
    std::filesystem::path p(get_executable_filename(argv0));

    return p.parent_path().parent_path().string();
}

} // namespace einsums::detail

namespace einsums::util {

namespace detail {

// CMake does not deal with explicit semicolons well, for this reason,
// the paths are delimited with ':'. On Windows those need to be
// converted to ';'.
std::string convert_delimiters(std::string paths) {
#if defined(EINSUMS_WINDOWS)
    std::replace(paths.begin(), paths.end(), ':', ';');
#endif
    return paths;
}

} // namespace detail

void runtime_configuration::pre_initialize_ini() {
    if (!need_to_call_pre_initialize)
        return;

    std::vector<std::string> lines = {
        // clang-format off
        // create an empty application section
        "[application]",

        // create system and application instance specific entries
        "[system]",
        "pid = " + std::to_string(getpid()),
#if defined(__linux) || defined(linux) || defined(__linux__)
        "executable_prefix = " + einsums::detail::get_executable_prefix(argv0),
#else
        "executable_prefix = " + einsums::detail::get_executable_prefix(),
#endif
        // create default installation location and logging settings
        "[einsums]",
        "master_ini_path = $[system.executable_prefix]/",
        "master_init_path_suffixes = /share/einsums:/../share/einsums",

        "install_signal_handlers = ${EINSUMS_INSTALL_SIGNAL_HANDLERS:0}",
        "diagnostics_on_terminate = ${EINSUMS_DIAGNOSTICS_ON_TERMINATE:1}",
        "attach_debugger = ${EINSUMS_ATTACH_DEBUGGER:0}",
        "exception_verbosity = ${EINSUMS_EXCEPTION_VERBOSITY:1}",

        // clang-format on
    };

    lines.insert(lines.end(), extra_static_ini_defs.begin(), extra_static_ini_defs.end());
    parse("<static defaults>", lines, false, false, false);

    need_to_call_pre_initialize = false;
}

void runtime_configuration::post_initialize_ini(std::string &einsums_ini_file_, std::vector<std::string> const &cmdline_ini_defs_) {
    util::init_ini_data_base(*this, einsums_ini_file_);
    need_to_call_pre_initialize = true;

    // let the command line override the config file.
    if (!cmdline_ini_defs_.empty()) {
        // do not weed out comments
        parse("<command line definitions>", cmdline_ini_defs_, true, false);
        need_to_call_pre_initialize = true;
    }
}

void runtime_configuration::pre_initialize_logging_ini() {
    std::vector<std::string> lines = {
        "[einsums.log]",
        "level = ${EINSUMS_LOG_LEVEL:3}",
        "destination = ${EINSUMS_LOG_DESTINATION:cerr}",
        "format = ${EINSUMS_LOG_FORMAT:"
        "[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%l%$] [host:%j] [pid:%P] [tid:%t] "
        "[pool:%w] [parent:%q] [task:%k] [%s:%#/%!] %v"
        "}",
    };

    // don't overload user overrides
    parse("<static logging defaults>", lines, false, false);
}

runtime_configuration::runtime_configuration(char const *argv0_, std::vector<std::string> const &extra_static_ini_defs_)
    : extra_static_ini_defs(extra_static_ini_defs_), need_to_call_pre_initialize(true)
#if defined(__linux) || defined(linux) || defined(__linux__)
      ,
      argv0(argv0_)
#endif
{
#if !defined(__linux) || defined(linux) || defined(__linux__)
    EINSUMS_UNUSED(argv0_);
#endif

    pre_initialize_ini();
}

void runtime_configuration::reconfigure(std::string const &einsums_ini_file_) {
    einsums_ini_file = einsums_ini_file_;
    reconfigure();
}

void runtime_configuration::reconfigure(const std::vector<std::string> &ini_defs) {
    cmdline_ini_defs = ini_defs;
    reconfigure();
}

void runtime_configuration::reconfigure() {
    pre_initialize_ini();
    pre_initialize_logging_ini();
    post_initialize_ini(einsums_ini_file, cmdline_ini_defs);
}

bool runtime_configuration::load_application_configuration(const char *filename, error_code &ec) {
    try {
        section appcfg(filename);
        section applroot;
        applroot.add_section("application", appcfg);
        section::merge(applroot);
    } catch (einsums::exception const &e) {
        if (&ec == &throws)
            throw;
        ec = make_error_code(e.get_error(), e.what(), throw_mode::rethrow);
        return false;
    }
    return true;
}

} // namespace einsums::util