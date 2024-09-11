//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/debugging/attach_debugger.hpp>
#include <einsums/runtime/config_entry.hpp>
#include <einsums/runtime/debugging.hpp>

#include <string>

namespace einsums::detail {

void may_attach_debugger(std::string const &category) {
    if (get_config_entry("einsums.attach_debugger", "") == category) {
        debug::detail::attach_debugger();
    }
}

} // namespace einsums::detail