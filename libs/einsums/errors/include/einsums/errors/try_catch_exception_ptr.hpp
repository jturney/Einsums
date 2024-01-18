//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <exception>

namespace einsums::detail {
/// Helper function for a try-catch block where what would normally go in
/// the catch block should be called after the catch block. This is useful
/// for situations where the catch-block may yield, since the catch block
/// should be started and ended on the same worker thread (with yielding and
/// stealing, the catch block may end on a different worker thread than
/// where it was started). Because of this, the helper's catch block only
/// stores the exception pointer, and forwards it outside the catch block.
///
/// Do not replace uses of try_catch_exception_ptr with a plain try-catch
/// without ensuring that the catch-block can never yield.
///
/// Note: Windows does not seem to have problems resuming a catch block on a
/// different worker thread, but we use this nonetheless on Windows since it
/// doesn't hurt.
template <typename TryCallable, typename CatchCallable>
EINSUMS_FORCEINLINE auto try_catch_exception_ptr(TryCallable &&t, CatchCallable &&c) -> decltype(auto) {
    std::exception_ptr ep;
    try {
        return t();
    } catch (...) {
        ep = std::current_exception();
    }
    return c(EINSUMS_MOVE(ep));
}
} // namespace einsums::detail
