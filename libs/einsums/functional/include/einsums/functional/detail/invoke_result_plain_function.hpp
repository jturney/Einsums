//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <utility>

namespace einsums::detail {

template <typename F, typename... Args>
using invoke_result_plain_function_t = decltype(std::declval<F>()(std::declval<Args>()...));

}