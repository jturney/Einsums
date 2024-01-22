//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

namespace einsums::threads::coroutines::detail {

struct default_hint {};
struct yield_hint : public default_hint {};
struct invoke_hint : public default_hint {};

/////////////////////////////////////////////////////////////////////////////
// This is the base class of all context implementations
struct context_impl_base {};

} // namespace einsums::threads::coroutines::detail