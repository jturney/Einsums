//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/modules/errors.hpp>
#include <einsums/runtime/custom_exception_info.hpp>
#include <einsums/runtime/debugging.hpp>
#include <einsums/runtime/runtime_handlers.hpp>

#include <iostream>
#include <sstream>
#include <string>

namespace einsums::detail {

[[noreturn]] void assertion_handler(std::source_location const &loc, const char *expr, std::string const &msg) {
    static thread_local bool handling_assertion = false;

    if (handling_assertion) {
        std::ostringstream strm;
        strm << "Trying to handle failed assertion while handling another failed assertion!" << std::endl;
        strm << "Assertion '" << expr << "' failed";
        if (!msg.empty()) {
            strm << " (" << msg << ")";
        }

        strm << std::endl;
        strm << "{file}: " << loc.file_name() << std::endl;
        strm << "{line}: " << loc.line() << std::endl;
        strm << "{function}: " << loc.function_name() << std::endl;

        std::cerr << strm.str();

        std::abort();
    }

    handling_assertion = true;

    std::ostringstream strm;
    strm << "Assertion '" << expr << "' failed";
    if (!msg.empty()) {
        strm << " (" << msg << ")";
    }

    einsums::exception e(einsums::error::assertion_failure, strm.str());
    std::cerr << einsums::detail::diagnostic_information(einsums::detail::get_exception(e, loc)) << std::endl;

    einsums::detail::may_attach_debugger("exception");

    std::abort();
}

} // namespace einsums::detail