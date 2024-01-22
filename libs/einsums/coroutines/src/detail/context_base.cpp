//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/coroutines/detail/context_base.hpp>
#include <einsums/coroutines/detail/coroutine_impl.hpp>

namespace einsums::threads::coroutines::detail {
template struct context_base<coroutine_impl>;
}