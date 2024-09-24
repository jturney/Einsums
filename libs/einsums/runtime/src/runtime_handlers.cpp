//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------
#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/debugging/backtrace.hpp>
#include <einsums/logging.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/modules/thread_manager.hpp>
#include <einsums/runtime/config_entry.hpp>
#include <einsums/runtime/custom_exception_info.hpp>
#include <einsums/runtime/debugging.hpp>
#include <einsums/runtime/runtime.hpp>
#include <einsums/runtime/runtime_handlers.hpp>
#include <einsums/threading_base/thread_data.hpp>
#include <einsums/threading_base/thread_pool_base.hpp>

#include <cstddef>
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

#if defined(EINSUMS_HAVE_VERIFY_LOCKS)
void registered_locks_error_handler() {
    std::string back_trace = einsums::debug::detail::trace(std::size_t(128));

    // throw or log, depending on config options
    if (get_config_entry("einsums.throw_on_held_lock", "1") == "0") {
        if (back_trace.empty()) {
            EINSUMS_LOG(debug, "suspending thread while at least one lock is being held "
                               "(stack backtrace was disabled at compile time)");
        } else {
            EINSUMS_LOG(debug, "suspending thread while at least one lock is being held, stack backtrace: {}", back_trace);
        }
    } else {
        if (back_trace.empty()) {
            EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status,
                                    "suspending thread while at least one lock is being held (stack backtrace was "
                                    "disabled at compile time)");
        } else {
            EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status,
                                    "suspending thread while at least one lock is being held, stack backtrace: {}", back_trace);
        }
    }
}

bool register_locks_predicate() {
    return einsums::threads::detail::get_self_ptr() != nullptr;
}
#endif

einsums::threads::detail::thread_pool_base *get_default_pool() {
    einsums::detail::runtime *rt = get_runtime_ptr();
    if (rt == nullptr) {
        EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status, "The runtime system is not active");
    }

    return &rt->get_thread_manager().default_pool();
}

einsums::threads::detail::mask_cref_type get_pu_mask(einsums::threads::detail::topology & /* topo */, std::size_t thread_num) {
    auto &rp = einsums::resource::get_partitioner();
    return rp.get_pu_mask(thread_num);
}
} // namespace einsums::detail
