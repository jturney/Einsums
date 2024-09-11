//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/util/manage_config.hpp>

#include <string>
#include <vector>

namespace einsums::detail {
///////////////////////////////////////////////////////////////////////////
inline std::string trim_whitespace(std::string const &s) {
    using size_type = std::string::size_type;

    size_type first = s.find_first_not_of(" \t");
    if (std::string::npos == first)
        return std::string();

    size_type last = s.find_last_not_of(" \t");
    return s.substr(first, last - first + 1);
}

///////////////////////////////////////////////////////////////////////////
manage_config::manage_config(std::vector<std::string> const &cfg) {
    add(cfg);
}

void manage_config::add(std::vector<std::string> const &cfg) {
    for (std::string const &s : cfg) {
        std::string::size_type p = s.find_first_of('=');
        std::string            key(trim_whitespace(s.substr(0, p)));
        if (key[key.size() - 1] == '!')
            key.erase(key.size() - 1);

        std::string value(trim_whitespace(s.substr(p + 1)));
        config_.insert(map_type::value_type(key, value));
    }
}
} // namespace einsums::detail
