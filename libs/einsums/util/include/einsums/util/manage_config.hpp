//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/string_util/from_string.hpp>

#include <map>
#include <string>
#include <vector>

namespace einsums::detail {
struct EINSUMS_EXPORT manage_config {
    using map_type = std::map<std::string, std::string>;

    manage_config(std::vector<std::string> const &cfg);

    void add(std::vector<std::string> const &cfg);

    template <typename T>
    T get_value(std::string const &key, T dflt = T()) const {
        map_type::const_iterator it = config_.find(key);
        if (it != config_.end())
            return einsums::string_util::from_string<T>((*it).second, dflt);
        return dflt;
    }

    map_type config_;
};
} // namespace einsums::detail
