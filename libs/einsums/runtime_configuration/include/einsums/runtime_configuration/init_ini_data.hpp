//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/ini/ini.hpp>

#include <string>

namespace einsums::util {

bool handle_ini_file(einsums::detail::section &ini, std::string const &loc);
bool handle_ini_file_env(einsums::detail::section &ini, char const *env_var, char const *file_suffix = nullptr);

///////////////////////////////////////////////////////////////////////////
// read system and user specified ini files
//
// returns true if at least one alternative location has been read
// successfully
bool init_ini_data_base(einsums::detail::section &ini, std::string &pika_ini_file);

///////////////////////////////////////////////////////////////////////////
// global function to read component ini information
void merge_component_inis(einsums::detail::section &ini);

} // namespace einsums::util