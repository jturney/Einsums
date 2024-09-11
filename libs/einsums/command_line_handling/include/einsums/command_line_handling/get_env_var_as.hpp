//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/export_definitions.hpp>
#include <einsums/logging.hpp>

#include <cstdlib>
#include <sstream>
#include <string>

namespace einsums::detail {

/// from env var name 's' get value if well-formed, otherwise return default
template <typename T>
T get_env_var_as(const char *s, T def) noexcept {
    T     val = def;
    char *env = std::getenv(s);
    if (env) {
        try {
            std::istringstream temp(env);
            temp >> val;
        } catch (...) {
            val = def;
            EINSUMS_LOG(err, "get_env_var_as - invalid {} {}", s, val);
        }
        EINSUMS_LOG(trace, "get_env_var_as {} {}", s, val);
    }
    return val;
}

} // namespace einsums::detail
