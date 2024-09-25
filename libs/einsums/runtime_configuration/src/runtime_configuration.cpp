//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/assert.hpp>
#include <einsums/errors/exception.hpp>
#include <einsums/filesystem.hpp>
// #include <einsums/itt_notify.hpp>
#include <einsums/preprocessor/expand.hpp>
#include <einsums/preprocessor/stringize.hpp>
#include <einsums/runtime_configuration/init_ini_data.hpp>
#include <einsums/runtime_configuration/runtime_configuration.hpp>
// #include <einsums/string_util/classification.hpp>
#include <einsums/string_util/from_string.hpp>
// #include <einsums/string_util/split.hpp>
#include <einsums/type_support/unused.hpp>
#include <einsums/util/get_entry_as.hpp>
#include <einsums/version.hpp>

#include <algorithm>
#include <boost/tokenizer.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
// #include <iterator>
#include <limits>
#include <map>
// #include <memory>
#include <set>
#include <string>
#include <system_error>
// #include <utility>
#include <vector>

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

            einsums::detail::split(path_dirs, epath, einsums::detail::is_any_of(":"), einsums::detail::token_compress_mode::on);

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

    EINSUMS_THROW_EXCEPTION(einsums::error::dynamic_link_failure, "get_executable_filename", "unable to find executable filename");

#elif defined(__APPLE__)
    EINSUMS_UNUSED(argv0);

    char          exe_path[PATH_MAX + 1];
    std::uint32_t len = sizeof(exe_path) / sizeof(exe_path[0]);

    if (0 != _NSGetExecutablePath(exe_path, &len)) {
        EINSUMS_THROW_EXCEPTION(einsums::error::dynamic_link_failure, "get_executable_filename", "unable to find executable filename");
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

// pre-initialize entries with compile time based values
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
            // NOLINTNEXTLINE(bugprone-suspicious-missing-comma)
            "master_ini_path_suffixes = /share/einsums" EINSUMS_INI_PATH_DELIMITER
                "/../share/einsums",
#ifdef EINSUMS_HAVE_ITTNOTIFY
            "use_itt_notify = ${EINSUMS_HAVE_ITTNOTIFY:0}",
#endif
            "shutdown_check_count = ${EINSUMS_SHUTDOWN_CHECK_COUNT:10}",
#ifdef EINSUMS_HAVE_VERIFY_LOCKS
#if defined(EINSUMS_DEBUG)
            "lock_detection = ${EINSUMS_LOCK_DETECTION:1}",
#else
            "lock_detection = ${EINSUMS_LOCK_DETECTION:0}",
#endif
            "throw_on_held_lock = ${EINSUMS_THROW_ON_HELD_LOCK:1}",
#endif
#ifdef EINSUMS_HAVE_THREAD_DEADLOCK_DETECTION
#ifdef EINSUMS_DEBUG
            "deadlock_detection = ${EINSUMS_DEADLOCK_DETECTION:1}",
#else
            "deadlock_detection = ${EINSUMS_DEADLOCK_DETECTION:0}",
#endif
#endif
#ifdef EINSUMS_HAVE_SPINLOCK_DEADLOCK_DETECTION
#ifdef EINSUMS_DEBUG
            "spinlock_deadlock_detection = "
            "${EINSUMS_SPINLOCK_DEADLOCK_DETECTION:1}",
#else
            "spinlock_deadlock_detection = "
            "${EINSUMS_SPINLOCK_DEADLOCK_DETECTION:0}",
#endif
            "spinlock_deadlock_detection_limit = "
            "${EINSUMS_SPINLOCK_DEADLOCK_DETECTION_LIMIT:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_SPINLOCK_DEADLOCK_DETECTION_LIMIT)) "}",
            "spinlock_deadlock_warning_limit = "
            "${EINSUMS_SPINLOCK_DEADLOCK_WARNING_LIMIT:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_SPINLOCK_DEADLOCK_WARNING_LIMIT)) "}",
#endif

            // add placeholders for keys to be added by command line handling
            "ignore_process_mask = 0",
            "process_mask = ${EINSUMS_PROCESS_MASK:}",
            "os_threads = cores",
            "cores = all",
            "first_pu = 0",
            "scheduler = local-priority-fifo",
            "affinity = core",
            "pu_step = 1",
            "pu_offset = 0",
            "numa_sensitive = 0",
            "max_idle_loop_count = ${EINSUMS_MAX_IDLE_LOOP_COUNT:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_IDLE_LOOP_COUNT_MAX)) "}",
            "max_busy_loop_count = ${EINSUMS_MAX_BUSY_LOOP_COUNT:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_BUSY_LOOP_COUNT_MAX)) "}",
#if defined(EINSUMS_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
            "max_idle_backoff_time = "
            "${EINSUMS_MAX_IDLE_BACKOFF_TIME:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_IDLE_BACKOFF_TIME_MAX)) "}",
#endif
            "default_scheduler_mode = ${EINSUMS_DEFAULT_SCHEDULER_MODE}",

            "install_signal_handlers = ${EINSUMS_INSTALL_SIGNAL_HANDLERS:0}",
            "diagnostics_on_terminate = ${EINSUMS_DIAGNOSTICS_ON_TERMINATE:1}",
            "attach_debugger = ${EINSUMS_ATTACH_DEBUGGER:0}",
            "exception_verbosity = ${EINSUMS_EXCEPTION_VERBOSITY:1}",
            "trace_depth = ${EINSUMS_TRACE_DEPTH:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH)) "}",

            "[einsums.stacks]",
            "small_size = ${EINSUMS_SMALL_STACK_SIZE:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_SMALL_STACK_SIZE)) "}",
            "medium_size = ${EINSUMS_MEDIUM_STACK_SIZE:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_MEDIUM_STACK_SIZE)) "}",
            "large_size = ${EINSUMS_LARGE_STACK_SIZE:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_LARGE_STACK_SIZE)) "}",
            "huge_size = ${EINSUMS_HUGE_STACK_SIZE:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_HUGE_STACK_SIZE)) "}",
#if defined(__linux) || defined(linux) || defined(__linux__) ||                \
    defined(__FreeBSD__)
            "use_guard_pages = ${EINSUMS_USE_GUARD_PAGES:0}",
#endif

            "[einsums.thread_queue]",
            "max_thread_count = ${EINSUMS_THREAD_QUEUE_MAX_THREAD_COUNT:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_THREAD_QUEUE_MAX_THREAD_COUNT)) "}",
            "min_tasks_to_steal_pending = "
            "${EINSUMS_THREAD_QUEUE_MIN_TASKS_TO_STEAL_PENDING:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_THREAD_QUEUE_MIN_TASKS_TO_STEAL_PENDING)) "}",
            "min_tasks_to_steal_staged = "
            "${EINSUMS_THREAD_QUEUE_MIN_TASKS_TO_STEAL_STAGED:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_THREAD_QUEUE_MIN_TASKS_TO_STEAL_STAGED)) "}",
            "min_add_new_count = "
            "${EINSUMS_THREAD_QUEUE_MIN_ADD_NEW_COUNT:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_THREAD_QUEUE_MIN_ADD_NEW_COUNT)) "}",
            "max_add_new_count = "
            "${EINSUMS_THREAD_QUEUE_MAX_ADD_NEW_COUNT:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_THREAD_QUEUE_MAX_ADD_NEW_COUNT)) "}",
            "min_delete_count = "
            "${EINSUMS_THREAD_QUEUE_MIN_DELETE_COUNT:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_THREAD_QUEUE_MIN_DELETE_COUNT)) "}",
            "max_delete_count = "
            "${EINSUMS_THREAD_QUEUE_MAX_DELETE_COUNT:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_THREAD_QUEUE_MAX_THREAD_COUNT)) "}",
            "max_terminated_threads = "
            "${EINSUMS_THREAD_QUEUE_MAX_TERMINATED_THREADS:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_THREAD_QUEUE_MAX_TERMINATED_THREADS)) "}",
            "init_threads_count = "
            "${EINSUMS_THREAD_QUEUE_INIT_THREADS_COUNT:" EINSUMS_PP_STRINGIZE(
                EINSUMS_PP_EXPAND(EINSUMS_THREAD_QUEUE_INIT_THREADS_COUNT)) "}",

#if defined(EINSUMS_HAVE_MPI)
            "[einsums.mpi]",
            "completion_mode = ${EINSUMS_MPI_COMPLETION_MODE:0}",
#endif

            "[einsums.commandline]",

            // allow for unknown options to be passed through
            "allow_unknown = ${EINSUMS_COMMANDLINE_ALLOW_UNKNOWN:0}",

            // allow for command line options to be passed through the
            // environment
            "prepend_options = ${EINSUMS_COMMANDLINE_OPTIONS}",
        // clang-format on
    };

    lines.insert(lines.end(), extra_static_ini_defs.begin(), extra_static_ini_defs.end());

    // don't overload user overrides
    this->parse("<static defaults>", lines, false, false, false);

    need_to_call_pre_initialize = false;
}

void runtime_configuration::post_initialize_ini(std::string &einsums_ini_file_, std::vector<std::string> const &cmdline_ini_defs_) {
    util::init_ini_data_base(*this, einsums_ini_file_);
    need_to_call_pre_initialize = true;

    // let the command line override the config file.
    if (!cmdline_ini_defs_.empty()) {
        // do not weed out comments
        this->parse("<command line definitions>", cmdline_ini_defs_, true, false);
        need_to_call_pre_initialize = true;
    }
}

void runtime_configuration::pre_initialize_logging_ini() {
    std::vector<std::string> lines = {
        "[einsums.log]",
        "level = ${EINSUMS_LOG_LEVEL:3}",
        "destination = ${EINSUMS_LOG_DESTINATION:cerr}",
        // "format = ${EINSUMS_LOG_FORMAT:[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%l%$] [host:%j] [pid:%P] [tid:%t] [pool:%w] [parent:%q] [task:%k]
        // [%s:%#/%!] %v"
        "format = ${EINSUMS_LOG_FORMAT:[%n] [%^%l%$] [pool:%w] [%s:%#/%!] %v"
        "}",
    };

    // don't overload user overrides
    this->parse("<static logging defaults>", lines, false, false);
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
runtime_configuration::runtime_configuration(char const *argv0_, std::vector<std::string> const &extra_static_ini_defs_)
    : extra_static_ini_defs(extra_static_ini_defs_), num_os_threads(0), small_stacksize(EINSUMS_SMALL_STACK_SIZE),
      medium_stacksize(EINSUMS_MEDIUM_STACK_SIZE), large_stacksize(EINSUMS_LARGE_STACK_SIZE), huge_stacksize(EINSUMS_HUGE_STACK_SIZE),
      need_to_call_pre_initialize(true)
#if defined(__linux) || defined(linux) || defined(__linux__)
      ,
      argv0(argv0_)
#endif
{
#if !(defined(__linux) || defined(linux) || defined(__linux__))
    EINSUMS_UNUSED(argv0_);
#endif
    pre_initialize_ini();

    // set global config options
#if EINSUMS_HAVE_ITTNOTIFY != 0
    use_ittnotify_api = get_itt_notify_mode();
#endif
    EINSUMS_ASSERT(init_small_stack_size() >= EINSUMS_SMALL_STACK_SIZE);

    small_stacksize  = init_small_stack_size();
    medium_stacksize = init_medium_stack_size();
    large_stacksize  = init_large_stack_size();
    EINSUMS_ASSERT(init_huge_stack_size() <= EINSUMS_HUGE_STACK_SIZE);
    huge_stacksize = init_huge_stack_size();
}

///////////////////////////////////////////////////////////////////////////
void runtime_configuration::reconfigure(std::string const &einsums_ini_file_) {
    einsums_ini_file = einsums_ini_file_;
    reconfigure();
}

void runtime_configuration::reconfigure(std::vector<std::string> const &cmdline_ini_defs_) {
    cmdline_ini_defs = cmdline_ini_defs_;
    reconfigure();
}

void runtime_configuration::reconfigure() {
    pre_initialize_ini();
    pre_initialize_logging_ini();
    post_initialize_ini(einsums_ini_file, cmdline_ini_defs);

    // set global config options
#if EINSUMS_HAVE_ITTNOTIFY != 0
    use_ittnotify_api = get_itt_notify_mode();
#endif
    EINSUMS_ASSERT(init_small_stack_size() >= EINSUMS_SMALL_STACK_SIZE);

    small_stacksize  = init_small_stack_size();
    medium_stacksize = init_medium_stack_size();
    large_stacksize  = init_large_stack_size();
    huge_stacksize   = init_huge_stack_size();
}

bool runtime_configuration::get_itt_notify_mode() const {
#if EINSUMS_HAVE_ITTNOTIFY != 0
    if (einsums::detail::section const *sec = get_section("einsums"); nullptr != sec) {
        return einsums::detail::get_entry_as<int>(*sec, "use_itt_notify", 0) != 0;
    }
#endif
    return false;
}

// Enable lock detection during suspension
bool runtime_configuration::enable_lock_detection() const {
#ifdef EINSUMS_HAVE_VERIFY_LOCKS
    if (einsums::detail::section const *sec = get_section("einsums"); nullptr != sec) {
        return einsums::detail::get_entry_as<int>(*sec, "lock_detection", 0) != 0;
    }
#endif
    return false;
}

// Enable minimal deadlock detection for einsums threads
bool runtime_configuration::enable_deadlock_detection() const {
#ifdef EINSUMS_HAVE_THREAD_DEADLOCK_DETECTION
    if (einsums::detail::section const *sec = get_section("einsums"); nullptr != sec) {
#    ifdef EINSUMS_DEBUG
        return einsums::detail::get_entry_as<int>(*sec, "deadlock_detection", 1) != 0;
#    else
        return einsums::detail::get_entry_as<int>(*sec, "deadlock_detection", 0) != 0;
#    endif
    }

#    ifdef EINSUMS_DEBUG
    return true;
#    else
    return false;
#    endif

#else
    return false;
#endif
}

///////////////////////////////////////////////////////////////////////////
bool runtime_configuration::enable_spinlock_deadlock_detection() const {
#ifdef EINSUMS_HAVE_SPINLOCK_DEADLOCK_DETECTION
    if (einsums::detail::section const *sec = get_section("einsums"); nullptr != sec) {
#    ifdef EINSUMS_DEBUG
        return einsums::detail::get_entry_as<int>(*sec, "spinlock_deadlock_detection", 1) != 0;
#    else
        return einsums::detail::get_entry_as<int>(*sec, "spinlock_deadlock_detection", 0) != 0;
#    endif
    }

#    ifdef EINSUMS_DEBUG
    return true;
#    else
    return false;
#    endif

#else
    return false;
#endif
}

///////////////////////////////////////////////////////////////////////////
std::size_t runtime_configuration::get_spinlock_deadlock_detection_limit() const {
#ifdef EINSUMS_HAVE_SPINLOCK_DEADLOCK_DETECTION
    if (einsums::detail::section const *sec = get_section("einsums"); nullptr != sec) {
        return einsums::detail::get_entry_as<std::size_t>(*sec, "spinlock_deadlock_detection_limit",
                                                          EINSUMS_SPINLOCK_DEADLOCK_DETECTION_LIMIT);
    }
    return EINSUMS_SPINLOCK_DEADLOCK_DETECTION_LIMIT;
#else
    return std::size_t(-1);
#endif
}

std::size_t runtime_configuration::get_spinlock_deadlock_warning_limit() const {
#ifdef EINSUMS_HAVE_SPINLOCK_DEADLOCK_DETECTION
    if (einsums::detail::section const *sec = get_section("einsums"); nullptr != sec) {
        return einsums::detail::get_entry_as<std::size_t>(*sec, "spinlock_deadlock_warning_limit", EINSUMS_SPINLOCK_DEADLOCK_WARNING_LIMIT);
    }
    return EINSUMS_SPINLOCK_DEADLOCK_WARNING_LIMIT;
#else
    return std::size_t(-1);
#endif
}

std::size_t runtime_configuration::trace_depth() const {
    if (einsums::detail::section const *sec = get_section("einsums"); nullptr != sec) {
        return einsums::detail::get_entry_as<std::size_t>(*sec, "trace_depth", EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH);
    }
    return EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH;
}

std::size_t runtime_configuration::get_os_thread_count() const {
    if (num_os_threads == 0) {
        num_os_threads = 1;
        if (einsums::detail::section const *sec = get_section("einsums"); nullptr != sec) {
            num_os_threads = einsums::detail::get_entry_as<std::uint32_t>(*sec, "os_threads", 1);
        }
    }
    return static_cast<std::size_t>(num_os_threads);
}

std::string runtime_configuration::get_cmd_line() const {
    if (einsums::detail::section const *sec = get_section("einsums"); nullptr != sec) {
        return sec->get_entry("cmd_line", "");
    }
    return "";
}

// Return the configured sizes of any of the know thread pools
std::size_t runtime_configuration::get_thread_pool_size(char const *poolname) const {
    if (einsums::detail::section const *sec = get_section("einsums.threadpools"); nullptr != sec) {
        return einsums::detail::get_entry_as<std::size_t>(*sec, std::string(poolname) + "_size", 2);
    }
    return 2; // the default size for all pools is 2
}

// Will return the stack size to use for all einsums-threads.
std::ptrdiff_t runtime_configuration::init_stack_size(char const *entryname, char const *defaultvaluestr,
                                                      std::ptrdiff_t defaultvalue) const {
    if (einsums::detail::section const *sec = get_section("einsums.stacks"); nullptr != sec) {
        std::string    entry  = sec->get_entry(entryname, defaultvaluestr);
        char          *endptr = nullptr;
        std::ptrdiff_t val    = std::strtoll(entry.c_str(), &endptr, /*base:*/ 0);
        return endptr != entry.c_str() ? val : defaultvalue;
    }
    return defaultvalue;
}

#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
bool runtime_configuration::use_stack_guard_pages() const {
    if (einsums::detail::section const *sec = get_section("einsums.stacks"); nullptr != sec) {
        return einsums::detail::get_entry_as<int>(*sec, "use_guard_pages", 1) != 0;
    }
    return true; // default is true
}
#endif

std::ptrdiff_t runtime_configuration::init_small_stack_size() const {
    return init_stack_size("small_size", EINSUMS_PP_STRINGIZE(EINSUMS_SMALL_STACK_SIZE), EINSUMS_SMALL_STACK_SIZE);
}

std::ptrdiff_t runtime_configuration::init_medium_stack_size() const {
    return init_stack_size("medium_size", EINSUMS_PP_STRINGIZE(EINSUMS_MEDIUM_STACK_SIZE), EINSUMS_MEDIUM_STACK_SIZE);
}

std::ptrdiff_t runtime_configuration::init_large_stack_size() const {
    return init_stack_size("large_size", EINSUMS_PP_STRINGIZE(EINSUMS_LARGE_STACK_SIZE), EINSUMS_LARGE_STACK_SIZE);
}

std::ptrdiff_t runtime_configuration::init_huge_stack_size() const {
    return init_stack_size("huge_size", EINSUMS_PP_STRINGIZE(EINSUMS_HUGE_STACK_SIZE), EINSUMS_HUGE_STACK_SIZE);
}

///////////////////////////////////////////////////////////////////////////
bool runtime_configuration::load_application_configuration(char const *filename, error_code &ec) {
    try {
        section appcfg(filename);
        section applroot;
        applroot.add_section("application", appcfg);
        this->section::merge(applroot);
    } catch (einsums::exception const &e) {
        // file doesn't exist or is ill-formed
        if (&ec == &throws)
            throw;
        ec = make_error_code(e.get_error(), e.what(), einsums::throw_mode::rethrow);
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////
std::ptrdiff_t runtime_configuration::get_stack_size(execution::thread_stacksize stacksize) const {
    switch (stacksize) {
    case execution::thread_stacksize::medium:
        return medium_stacksize;

    case execution::thread_stacksize::large:
        return large_stacksize;

    case execution::thread_stacksize::huge:
        return huge_stacksize;

    case execution::thread_stacksize::nostack:
        return (std::numeric_limits<std::ptrdiff_t>::max)();

    default:
    case execution::thread_stacksize::small_:
        break;
    }
    return small_stacksize;
}
} // namespace einsums::util
