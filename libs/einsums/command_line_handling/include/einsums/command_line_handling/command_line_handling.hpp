//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/program_options/options_description.hpp>
#include <einsums/program_options/variables_map.hpp>
#include <einsums/runtime_configuration/runtime_configuration.hpp>
#include <einsums/util/manage_config.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace einsums::detail {
enum class command_line_handling_result {
    success, // All went well, continue starting the runtime
    exit,    // All went well, but should exit (e.g. --einsums:help was given)
};

struct command_line_handling {
    command_line_handling(einsums::util::runtime_configuration rtcfg, std::vector<std::string> ini_config,
                          std::function<int(einsums::program_options::variables_map &vm)> einsums_main_f)
        : _rtcfg(rtcfg), _ini_config(ini_config)
          // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
          ,
          _einsums_main_f(einsums_main_f), _num_threads(1), _num_cores(1), _pu_step(1), _pu_offset(std::size_t(-1)), _numa_sensitive(0),
          _use_process_mask(true), _cmd_line_parsed(false), _info_printed(false), _version_printed(false) {}

    command_line_handling_result call(einsums::program_options::options_description const &desc_cmdline, int argc, const char *const *argv);

    einsums::program_options::variables_map _vm;
    einsums::util::runtime_configuration    _rtcfg;

    std::vector<std::string>                                        _ini_config;
    std::function<int(einsums::program_options::variables_map &vm)> _einsums_main_f;

    std::size_t _num_threads;
    std::size_t _num_cores;
    std::size_t _pu_step;
    std::size_t _pu_offset;
    std::string _queuing;
    std::string _affinity_domain;
    std::string _affinity_bind;
    std::size_t _numa_sensitive;
    bool        _use_process_mask;
    std::string _process_mask;
    bool        _cmd_line_parsed;
    bool        _info_printed;
    bool        _version_printed;

  protected:
    // Helper functions for checking command line options
    void check_affinity_domain() const;
    void check_affinity_description() const;
    void check_pu_offset() const;
    void check_pu_step() const;

    void handle_arguments(detail::manage_config &cfgmap, einsums::program_options::variables_map &_vm,
                          std::vector<std::string> &ini_config);

    void update_logging_settings(einsums::program_options::variables_map &vm, std::vector<std::string> &ini_config);

    void store_command_line(int argc, const char *const *argv);
    void store_unregistered_options(std::string const &cmd_name, std::vector<std::string> const &unregistered_options);
    bool handle_help_options(einsums::program_options::options_description const &help);

    void handle_attach_debugger();

    std::vector<std::string> preprocess_config_settings(int argc, const char *const *argv);
};
} // namespace einsums::detail
