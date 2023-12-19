//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/Assert.hpp>
#include <einsums/Config.hpp>
#include <iostream>
#include <string>

namespace einsums::detail {
assertion_handler_type &get_handler() {
    static assertion_handler_type handler = nullptr;
    return handler;
}

void set_assertion_handler(assertion_handler_type handler) {
    if (detail::get_handler() == nullptr) {
        detail::get_handler() = handler;
    }
}

void handle_assert(SourceLocation const &loc, const char *expr, std::string const &msg) noexcept {
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