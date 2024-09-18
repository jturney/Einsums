//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/threading_base/detail/get_default_pool.hpp>
#include <einsums/threading_base/scheduler_base.hpp>
#include <einsums/threading_base/thread_description.hpp>
#include <einsums/threading_base/thread_pool_base.hpp>

// The following implementation has been divided for Linux and Mac OSX
#if (defined(__linux) || defined(__linux__) || defined(linux) || defined(__APPLE__))

namespace einsums_start {
// Redefining weak variables defined in einsums_main.hpp to facilitate error
// checking and make sure correct errors are thrown. It is added again
// to make sure that these variables are defined correctly in cases
// where einsums_main functionalities are not used.
EINSUMS_SYMBOL_EXPORT bool is_linked __attribute__((weak))               = false;
EINSUMS_SYMBOL_EXPORT bool include_libeinsums_wrap __attribute__((weak)) = false;
} // namespace einsums_start

#endif

namespace einsums::threads::detail {
static get_default_pool_type get_default_pool;

void set_get_default_pool(get_default_pool_type f) {
    get_default_pool = f;
}

thread_pool_base *get_self_or_default_pool() {
    thread_pool_base *pool      = nullptr;
    auto              thrd_data = get_self_id_data();
    if (thrd_data) {
        pool = thrd_data->get_scheduler_base()->get_parent_pool();
    } else if (get_default_pool) {
        pool = get_default_pool();
        EINSUMS_ASSERT(pool);
    } else {
        EINSUMS_THROW_EXCEPTION(einsums::error::invalid_status,
                                "Attempting to register a thread outside the einsums runtime and no default pool "
                                "handler is installed. Did you mean to run this on a einsums thread?");
    }

    return pool;
}
} // namespace einsums::threads::detail
