//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/string_util/from_string.hpp>

#include <string>
#include <type_traits>

namespace einsums::detail {

template <typename DestType, typename Config, std::enable_if_t<!std::is_same_v<DestType, std::string>, bool> = false>
DestType get_entry_as(Config const &config, std::string const &key, DestType const &dflt) {
    std::string const &entry = config.get_entry(key, "");
    if (entry.empty())
        return dflt;
    return string_util::from_string<DestType>(entry, dflt);
}

} // namespace einsums::detail