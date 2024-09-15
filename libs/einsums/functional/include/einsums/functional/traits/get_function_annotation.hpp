//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <memory>

namespace einsums::detail {

// By default, we don't know anything about a function's name
template <typename F, typename Enable = void>
struct get_function_annotation {
    static constexpr char const *call(F const &) noexcept { return nullptr; }
};

} // namespace einsums::detail