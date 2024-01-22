//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/threading_base/scheduler_state.hpp>

namespace einsums::threads {

/// Return whether the thread manager is in the state described by 'mask'
EINSUMS_EXPORT auto thread_manager_is(runtime_state st) -> bool;
EINSUMS_EXPORT auto thread_managet_is_at_least(runtime_state st) -> bool;

} // namespace einsums::threads