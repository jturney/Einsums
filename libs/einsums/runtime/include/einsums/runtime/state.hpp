//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/runtime/state.hpp>
#include <einsums/threading_base/scheduler_state.hpp>

namespace einsums::detail {
// return whether thread manager is in the state described by 'mask'
EINSUMS_EXPORT bool thread_manager_is(runtime_state st);
} // namespace einsums::detail
