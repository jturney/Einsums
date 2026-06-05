//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Debugging/Backtrace.hpp>
#include <Einsums/Version.hpp>

#include <iostream>

namespace einsums::detail {

namespace {
auto get_handler() -> assertion_handler_type & {
    static assertion_handler_type handler{default_assertion_handler};
    return handler;
}
} // namespace

void default_assertion_handler(std::source_location const &loc, char const *expr, std::string const &msg) {
    std::ostringstream err_str;
    err_str << complete_version() << "\n" << loc.function_name() << ":" << loc.line() << " : Assertion '" << expr << "' failed";
    if (!msg.empty()) {
        err_str << " (" << msg << ")\n";
    } else {
        err_str << "\n";
    }

    err_str << "\n" << util::backtrace() << "\n";

    throw assertion_error(err_str.str());
}

void set_assertion_handler(assertion_handler_type handler) {
    get_handler() = handler;
}

void handle_assert(std::source_location const &loc, char const *expr, std::string const &msg) {
    if (get_handler() == nullptr) {
        default_assertion_handler(loc, expr, msg);
    } else {
        get_handler()(loc, expr, msg);
    }
}

} // namespace einsums::detail
