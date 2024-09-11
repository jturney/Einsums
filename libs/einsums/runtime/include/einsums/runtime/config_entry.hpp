//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <string>

namespace einsums::detail {

EINSUMS_EXPORT std::string get_config_entry(std::string const &key, std::string const &dflt);
EINSUMS_EXPORT std::string get_config_entry(std::string const &key, std::size_t dflt);

EINSUMS_EXPORT void set_config_entry(std::string const &key, std::string const &dflt);
EINSUMS_EXPORT void set_config_entry(std::string const &key, std::size_t dflt);

EINSUMS_EXPORT void set_config_entry_callback(std::string const                                                   &key,
                                              std::function<void(std::string const &, std::string const &)> const &callback);
} // namespace einsums::detail