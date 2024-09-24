//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/modules/thread_manager.hpp>
#include <einsums/runtime/runtime.hpp>
#include <einsums/runtime/state.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace einsums::detail {
// return whether thread manager is in the runtime_state described by st
bool thread_manager_is(einsums::runtime_state st) {
    einsums::detail::runtime *rt = get_runtime_ptr();
    if (nullptr == rt) {
        // we're probably either starting or stopping
        return st <= einsums::runtime_state::starting || st >= einsums::runtime_state::stopping;
    }
    return (rt->get_thread_manager().status() == st);
}
} // namespace einsums::detail
