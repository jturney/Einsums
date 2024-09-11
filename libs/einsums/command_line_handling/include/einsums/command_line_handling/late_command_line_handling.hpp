//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/program_options/options_description.hpp>
#include <einsums/runtime_configuration/runtime_configuration.hpp>

#include <cstddef>

namespace einsums::detail {
int handle_late_commandline_options(einsums::util::runtime_configuration &ini, einsums::program_options::options_description const &options,
                                    void (*handle_print_bind)(std::size_t));
} // namespace einsums::detail
