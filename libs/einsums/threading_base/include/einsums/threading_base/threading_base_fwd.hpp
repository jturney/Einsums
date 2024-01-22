//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/function.hpp>
#include <einsums/functional/unique_function.hpp>
#include <einsums/modules/errors.hpp>

#if defined(EINSUMS_HAVE_APEX)
#    include <apex_api.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

namespace einsums::detail::external_timer {
#if defined(EINSUMS_HAVE_APEX)
using apex::task_wrapper;
#else
struct task_wrapper {};
#endif
} // namespace einsums::detail::external_timer

namespace einsums::threads {
struct EINSUMS_EXPORT thread_pool_base;
}

namespace einsums::threads::detail {

/// \cond NOINTERNAL
struct scheduler_base;
struct thread_data;
struct thread_data_stackful;
struct thread_data_stackless;

using thread_id_ref_type = thread_id_ref;

} // namespace einsums::threads::detail