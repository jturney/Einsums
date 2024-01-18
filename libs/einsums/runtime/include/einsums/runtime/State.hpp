//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/threading_base/SchedulerState.hpp>

namespace einsums::threads {

/// Return whether the thread manager is in the state described by 'mask'
EINSUMS_EXPORT auto thread_manager_is(RuntimeState st) -> bool;
EINSUMS_EXPORT auto thread_managet_is_at_least(RuntimeState st) -> bool;

} // namespace einsums::threads