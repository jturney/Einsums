//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>

#include <iostream>

namespace einsums::detail {

namespace {
auto get_handler() -> assertion_handler_type & {
    static assertion_handler_type handler{nullptr};
    return handler;
}
} // namespace

void set_assertion_handler(assertion_handler_type handler) {
    if (get_handler() == nullptr) {
        get_handler() = handler;
    }
}

namespace detail {
void handle_assert(std::source_location const &loc, const char *expr, std::string const &msg) noexcept {
    if (get_handler() == nullptr) {
        std::cerr << loc.function_name() << ":" << loc.line() << " : Assertion '" << expr << "' failed";
        if (!msg.empty()) {
            std::cerr << " (" << msg << ")\n";
        } else {
            std::cerr << "\n";
        }
        std::abort();
    }
    get_handler()(loc, expr, msg);
}

} // namespace detail
} // namespace einsums::detail
