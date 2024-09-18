//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/function.hpp>
#include <einsums/threading_base/thread_pool_base.hpp>

namespace einsums::threads::detail {
using get_default_pool_type = util::detail::function<thread_pool_base *()>;
EINSUMS_EXPORT void              set_get_default_pool(get_default_pool_type f);
EINSUMS_EXPORT thread_pool_base *get_self_or_default_pool();
} // namespace einsums::threads::detail
