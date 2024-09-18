//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/coroutines/detail/coroutine_self.hpp>

#include <cstddef>

namespace einsums::threads::coroutines::detail {
coroutine_self *&coroutine_self::local_self() {
    static thread_local coroutine_self *local_self_ = nullptr;
    return local_self_;
}
} // namespace einsums::threads::coroutines::detail
