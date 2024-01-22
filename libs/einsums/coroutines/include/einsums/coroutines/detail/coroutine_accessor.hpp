//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

namespace einsums::threads::coroutines::detail {
struct coroutine_accessor {
    template <typename Coroutine>
    EINSUMS_FORCEINLINE static typename Coroutine::impl_ptr get_impl(Coroutine &x) {
        return x.get_impl();
    }
};
} // namespace einsums::threads::coroutines::detail
