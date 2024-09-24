//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/errors/error_code.hpp>
#include <einsums/ini/ini.hpp>
#include <einsums/runtime_configuration/runtime_configuration_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace einsums::util {
///////////////////////////////////////////////////////////////////////////
// The runtime_configuration class is a wrapper for the runtime
// configuration data allowing to extract configuration information in a
// more convenient way
class runtime_configuration : public einsums::detail::section {
    std::string              einsums_ini_file;
    std::vector<std::string> cmdline_ini_defs;
    std::vector<std::string> extra_static_ini_defs;

  public:
    // initialize and load configuration information
    explicit runtime_configuration(char const *argv0, std::vector<std::string> const &extra_static_ini_defs = {});

    // re-initialize all entries based on the additional information from
    // the given configuration file
    void reconfigure(std::string const &ini_file);

    // re-initialize all entries based on the additional information from
    // any explicit command line options
    void reconfigure(std::vector<std::string> const &ini_defs);

    // Load application specific configuration and merge it with the
    // default configuration loaded from einsums.ini
    bool load_application_configuration(char const *filename, error_code &ec = throws);

    // Can be set to true if we want to use the ITT notify tools API.
    bool get_itt_notify_mode() const;

    // Enable lock detection during suspension
    bool enable_lock_detection() const;

    // Enable global lock tracking
    bool enable_global_lock_detection() const;

    // Enable minimal deadlock detection for einsums threads
    bool        enable_deadlock_detection() const;
    bool        enable_spinlock_deadlock_detection() const;
    std::size_t get_spinlock_deadlock_detection_limit() const;
    std::size_t get_spinlock_deadlock_warning_limit() const;

#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
    bool use_stack_guard_pages() const;
#endif

    // return trace_depth for stack-backtraces
    std::size_t trace_depth() const;

    // Returns the number of OS threads this process is running.
    std::size_t get_os_thread_count() const;

    // Returns the command line that this process was invoked with.
    std::string get_cmd_line() const;

    // Will return the default stack size to use for all einsums-threads.
    std::ptrdiff_t get_default_stack_size() const { return small_stacksize; }

    // Will return the requested stack size to use for a einsums-threads.
    std::ptrdiff_t get_stack_size(execution::thread_stacksize stacksize) const;

    // Return the configured sizes of any of the know thread pools
    std::size_t get_thread_pool_size(char const *poolname) const;

  private:
    std::ptrdiff_t init_stack_size(char const *entryname, char const *defaultvaluestr, std::ptrdiff_t defaultvalue) const;

    std::ptrdiff_t init_small_stack_size() const;
    std::ptrdiff_t init_medium_stack_size() const;
    std::ptrdiff_t init_large_stack_size() const;
    std::ptrdiff_t init_huge_stack_size() const;

    void pre_initialize_ini();
    void post_initialize_ini(std::string &einsums_ini_file, std::vector<std::string> const &cmdline_ini_defs);
    void pre_initialize_logging_ini();

    void reconfigure();

  private:
    mutable std::uint32_t num_os_threads;
    std::ptrdiff_t        small_stacksize;
    std::ptrdiff_t        medium_stacksize;
    std::ptrdiff_t        large_stacksize;
    std::ptrdiff_t        huge_stacksize;
    bool                  need_to_call_pre_initialize;
#if defined(__linux) || defined(linux) || defined(__linux__)
    char const *argv0;
#endif
};
} // namespace einsums::util
