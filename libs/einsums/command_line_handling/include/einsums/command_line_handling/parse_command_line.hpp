//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/ini/ini.hpp>
#include <einsums/program_options/options_description.hpp>
#include <einsums/program_options/variables_map.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace einsums::detail {
enum class commandline_error_mode {
    return_on_error,
    rethrow_on_error,
    allow_unregistered,
    ignore_aliases             = 0x40,
    report_missing_config_file = 0x80
};

commandline_error_mode  operator&(commandline_error_mode const lhs, commandline_error_mode const rhs) noexcept;
commandline_error_mode &operator&=(commandline_error_mode &lhs, commandline_error_mode rhs) noexcept;
commandline_error_mode  operator|(commandline_error_mode const lhs, commandline_error_mode rhs) noexcept;
commandline_error_mode &operator|=(commandline_error_mode &lhs, commandline_error_mode rhs) noexcept;
commandline_error_mode  operator~(commandline_error_mode m) noexcept;
bool                    contains_error_mode(commandline_error_mode const m, commandline_error_mode const b) noexcept;
std::string             enquote(std::string const &arg);

void parse_commandline(einsums::detail::section const &rtcfg, einsums::program_options::options_description const &app_options,
                       std::string const &cmdline, einsums::program_options::variables_map &vm,
                       commandline_error_mode                         error_mode           = commandline_error_mode::return_on_error,
                       einsums::program_options::options_description *visible              = nullptr,
                       std::vector<std::string>                      *unregistered_options = nullptr);

void parse_commandline(einsums::detail::section const &rtcfg, einsums::program_options::options_description const &app_options,
                       std::string const &arg0, std::vector<std::string> const &args, einsums::program_options::variables_map &vm,
                       commandline_error_mode                         error_mode           = commandline_error_mode::return_on_error,
                       einsums::program_options::options_description *visible              = nullptr,
                       std::vector<std::string>                      *unregistered_options = nullptr);

std::string reconstruct_command_line(einsums::program_options::variables_map const &vm);
} // namespace einsums::detail
