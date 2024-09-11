//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <functional>
#include <type_traits>
#include <utility>

namespace einsums {

template <typename F, typename... Args>
auto bind_back(F func, Args &&...args) {
    return [func, ... bound_args = std::forward<Args>(args)](auto &&...call_args) mutable {
        return std::invoke(func, std::forward<decltype(call_args)>(call_args)..., bound_args...);
    };
}

} // namespace einsums