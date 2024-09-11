//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/modules/debugging.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/runtime/config_entry.hpp>
#include <einsums/runtime/custom_exception_info.hpp>
#include <einsums/runtime/debugging.hpp>
#include <einsums/runtime/runtime.hpp>
#include <einsums/version.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>

#if defined(EINSUMS_WINDOWS)
#    include <process.h>
#elif defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace einsums::detail {

// For testing purposes, we sometime expect to see exceptions, allow them to go
// through without attaching a debugger
std::atomic<bool> expect_exception_flag(false);

bool expect_exception(bool flag) {
    return expect_exception_flag.exchange(flag);
}

// Extract the diagnostic information embedded in the given exception and
// return a string holding a formatted message.
std::string diagnostic_information(einsums::exception_info const &xi) {
    int const verbosity = string_util::from_string<int>(get_config_entry("einsums.exception_verbosity", "1"));

    std::ostringstream strm;
    strm << "\n";

    if (verbosity >= 2) {
        strm << get_full_build_string();

        std::string const *env = xi.get<throw_env>();
        if (env && !env->empty()) {
            strm << "{env}: " << *env;
        }
    }

    std::string const *file = xi.get<throw_file>();
    if (file) {
        strm << "{file}: " << *file << "\n";
    }

    long const *line = xi.get<throw_line>();
    if (line) {
        strm << "{line}: " << *line << "\n";
    }

    std::string const *function = xi.get<throw_function>();
    if (function) {
        strm << "{function}: " << *function << "\n";
    }

    std::exception const *se = dynamic_cast<std::exception const *>(&xi);
    if (se) {
        strm << "{what}: " << se->what() << "\n";
    }

    return strm.str();
}

void pre_exception_handler() {
    if (!expect_exception_flag.load(std::memory_order_relaxed)) {
        may_attach_debugger("exception");
    }
}

static get_full_build_string_type get_full_build_string_f;

void set_get_full_build_string(get_full_build_string_type f) {
    get_full_build_string_f = f;
}

std::string get_full_build_string() {
    if (get_full_build_string_f) {
        return get_full_build_string_f();
    }
    return full_build_string();
}

void report_exception_and_continue(std::exception const &e) {
    pre_exception_handler();

    std::cerr << e.what() << std::endl;
}

void report_exception_and_continue(std::exception_ptr const &e) {
    pre_exception_handler();

    std::cerr << diagnostic_information(e) << std::endl;
}

void report_exception_and_continue(einsums::exception const &e) {
    pre_exception_handler();

    std::cerr << diagnostic_information(e) << std::endl;
}

void report_exception_and_terminate(std::exception const &e) {
    report_exception_and_continue(e);
    std::abort();
}

void report_exception_and_terminate(std::exception_ptr const &e) {
    report_exception_and_continue(e);
    std::abort();
}

void report_exception_and_terminate(einsums::exception const &e) {
    report_exception_and_continue(e);
    std::abort();
}

einsums::exception_info construct_exception_info(std::string const &func, std::string const &file, long line, std::string const &back_trace,
                                                 std::string const &hostname, std::int64_t pid, std::size_t shepherd, std::size_t thread_id,
                                                 std::string const &thread_name, std::string const &env, std::string const &config,
                                                 std::string const &state_name, std::string const &auxinfo) {
    return einsums::exception_info().set(
        einsums::detail::throw_stacktrace(back_trace), einsums::detail::throw_hostname(hostname), einsums::detail::throw_pid(pid),
        einsums::detail::throw_shepherd(shepherd), einsums::detail::throw_thread_id(thread_id),
        einsums::detail::throw_thread_name(thread_name), einsums::detail::throw_function(func), einsums::detail::throw_file(file),
        einsums::detail::throw_line(line), einsums::detail::throw_env(env), einsums::detail::throw_config(config),
        einsums::detail::throw_state(state_name), einsums::detail::throw_auxinfo(auxinfo));
}

template <typename Exception>
std::exception_ptr construct_exception(Exception const &e, einsums::exception_info info) {
    try {
        throw_with_info(e, std::move(info));
    } catch (...) {
        return std::current_exception();
    }

    return std::exception_ptr();
}

template EINSUMS_EXPORT std::exception_ptr construct_exception(einsums::exception const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(std::system_error const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(std::exception const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(einsums::detail::std_exception const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(std::bad_exception const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(einsums::detail::bad_exception const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(std::bad_typeid const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(einsums::detail::bad_typeid const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(std::bad_cast const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(einsums::detail::bad_cast const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(std::bad_alloc const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(einsums::detail::bad_alloc const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(std::logic_error const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(std::runtime_error const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(std::out_of_range const &, einsums::exception_info info);
template EINSUMS_EXPORT std::exception_ptr construct_exception(std::invalid_argument const &, einsums::exception_info info);

inline std::size_t get_arraylen(char **array) {
    std::size_t count = 0;
    if (array != nullptr) {
        while (array[count] != nullptr)
            ++count;
    }
    return count;
}

std::string get_execution_environment() {
    std::vector<std::string> env;

#if defined(EINSUMS_WINDOWS)
    std::size_t len = get_arraylen(_environ);
    env.reserve(len);
    std::copy(&_environ[0], &_environ[len], std::back_inserter(env));
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__AIX__)
    std::size_t len = get_arraylen(environ);
    env.reserve(len);
    std::copy(&environ[0], &environ[len], std::back_inserter(env));
#elif defined(__APPLE__)
    std::size_t len = get_arraylen(environ);
    env.reserve(len);
    std::copy(&environ[0], &environ[len], std::back_inserter(env));
#else
#    error "Don't know, how to access the execution environment on this platform"
#endif

    std::sort(env.begin(), env.end());

    static constexpr char const *ignored_env_patterns[] = {"DOCKER", "GITHUB_TOKEN"};
    std::string                  retval                 = fmt::format("{} entries:\n", env.size());
    for (std::string const &s : env) {
        if (std::all_of(std::begin(ignored_env_patterns), std::end(ignored_env_patterns),
                        [&s](auto const e) { return s.find(e) == std::string::npos; })) {
            retval += "  " + s + "\n";
        }
    }
    return retval;
}

einsums::exception_info custom_exception_info(std::string const &func, std::string const &file, long line, std::string const &auxinfo) {
    std::int64_t pid = ::getpid();

    // Add trace information?

    std::string env(get_execution_environment());
    std::string config(configuration_string());

    return einsums::exception_info().set(throw_function(func), throw_file(file), throw_line(line), throw_env(env), throw_config(config),
                                         throw_auxinfo(auxinfo));
}

std::string get_error_host_name(einsums::exception_info const &xi) {
    std::string const *hostname_ = xi.get<throw_hostname>();
    if (hostname_ && !hostname_->empty())
        return *hostname_;
    return {};
}

std::int64_t get_error_process_id(einsums::exception_info const &xi) {
    std::int64_t const *pid_ = xi.get<throw_pid>();
    if (pid_)
        return *pid_;
    return -1;
}

std::string get_error_env(einsums::exception_info const &xi) {
    std::string const *env = xi.get<throw_env>();
    if (env && !env->empty())
        return *env;

    return "<unknown>";
}

std::string get_error_config(einsums::exception_info const &xi) {
    std::string const *config_info = xi.get<throw_config>();
    if (config_info && !config_info->empty())
        return *config_info;
    return std::string();
}

} // namespace einsums::detail