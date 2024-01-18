//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>

#include <iostream>
#include <string>

namespace einsums::detail {
auto get_handler() -> assertion_handler_type & {
    static assertion_handler_type handler = nullptr;
    return handler;
}

void set_assertion_handler(assertion_handler_type handler) {
    if (detail::get_handler() == nullptr) {
        detail::get_handler() = handler;
    }
}

void handle_assert(source_location const &loc, char const *expr, std::string const &msg) noexcept {
    if (get_handler() == nullptr) {
        std::cerr << loc << ": Assertion '" << expr << "' failed";
        if (!msg.empty()) {
            std::cerr << " (" << msg << ")\n";
        } else {
            std::cerr << '\n';
        }
        std::abort();
    }
    get_handler()(loc, expr, msg);
}

} // namespace einsums::detail