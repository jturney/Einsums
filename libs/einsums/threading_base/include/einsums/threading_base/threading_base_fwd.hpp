//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/coroutines/coroutine_fwd.hpp>
#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/coroutines/thread_id_type.hpp>
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
using thread_id_type     = thread_id;

using coroutine_type           = coroutines::detail::coroutine;
using stackless_coroutine_type = coroutines::detail::stackless_coroutine;

using thread_result_type = std::pair<thread_schedule_state, thread_id_type>;
using thread_arg_type    = thread_restart_state;

using thread_function_sig  = thread_result_type(thread_arg_type);
using thread_function_type = util::detail::unique_function<thread_function_sig>;

using thread_self           = coroutines::detail::coroutine_self;
using thread_self_impl_type = coroutines::detail::coroutine_impl;

#if defined(EINSUMS_HAVE_APEX)
EINSUMS_EXPORT std::shared_ptr<einsums::detail::external_timer::task_wrapper> get_self_timer_data(void);
EINSUMS_EXPORT void set_self_timer_data(std::shared_ptr<einsums::detail::external_timer::task_wrapper> data);
#endif
/// \endcond
} // namespace einsums::threads::detail