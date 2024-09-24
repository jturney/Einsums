//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/threading_base/thread_pool_base.hpp>

#include <source_location>
#include <string>

namespace einsums::detail {

[[noreturn]] EINSUMS_EXPORT void assertion_handler(std::source_location const &loc, const char *expr, std::string const &msg);
#if defined(EINSUMS_HAVE_VERIFY_LOCKS)
EINSUMS_EXPORT void registered_locks_error_handler();
EINSUMS_EXPORT bool register_locks_predicate();
#endif
EINSUMS_EXPORT einsums::threads::detail::thread_pool_base *get_default_pool();
EINSUMS_EXPORT einsums::threads::detail::mask_cref_type get_pu_mask(einsums::threads::detail::topology &topo, std::size_t thread_num);

} // namespace einsums::detail