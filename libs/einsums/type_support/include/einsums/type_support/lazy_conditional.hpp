//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <type_traits>

namespace einsums::detail {
template <bool Enable, typename C1, typename C2>
struct lazy_conditional : std::conditional<Enable, C1, C2>::type {};

template <bool Enable, typename C1, typename C2>
using lazy_conditional_t = typename lazy_conditional<Enable, C1, C2>::type;
} // namespace einsums::type_support
