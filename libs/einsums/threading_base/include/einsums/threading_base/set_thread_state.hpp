//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/coroutines/coroutine.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/threading_base/threading_base_fwd.hpp>

#include <atomic>
#include <memory>

namespace einsums::threads::detail {
EINSUMS_EXPORT thread_state set_thread_state(thread_id_type const &id, thread_schedule_state new_state, thread_restart_state new_state_ex,
                                             execution::thread_priority      priority,
                                             execution::thread_schedule_hint schedulehint = execution::thread_schedule_hint(),
                                             bool retry_on_active = true, error_code &ec = throws);
} // namespace einsums::threads::detail
