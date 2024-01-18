//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

namespace einsums::detail {

///////////////////////////////////////////////////////////////////////////
// This utility simplifies templates returning compatible types
//
// Usage: return void_guard<Result>(), expr;
// - Result != void -> return expr;
// - Result == void -> return (void)expr;
template <typename Result>
struct void_guard {};

template <>
struct void_guard<void> {
    template <typename T>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE void operator,(T const &) const noexcept {}
};

} // namespace einsums::detail