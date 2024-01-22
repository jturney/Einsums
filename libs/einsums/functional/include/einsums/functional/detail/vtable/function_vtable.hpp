//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/detail/vtable/callable_vtable.hpp>
#include <einsums/functional/detail/vtable/copyable_vtable.hpp>
#include <einsums/functional/detail/vtable/vtable.hpp>

#include <type_traits>

namespace einsums::util::detail {

struct function_base_vtable : vtable, copyable_vtable, callable_info_vtable {
    template <typename T>
    constexpr function_base_vtable(construct_vtable<T>, std::integral_constant<bool, true>) noexcept
        : vtable(construct_vtable<T>()), copyable_vtable(construct_vtable<T>()),
          callable_info_vtable(construct_vtable<T>()) {}

    template <typename T>
    constexpr function_base_vtable(construct_vtable<T>, std::integral_constant<bool, false>) noexcept
        : vtable(construct_vtable<T>()), copyable_vtable(nullptr), callable_info_vtable(construct_vtable<T>()) {}
};

template <typename Sig, bool Copyable = true>
struct function_vtable;

template <typename Sig>
struct function_vtable<Sig, /*Copyable*/ false> : function_base_vtable, callable_vtable<Sig> {
    using copyable_tag = std::integral_constant<bool, false>;

    template <typename T>
    constexpr function_vtable(construct_vtable<T>) noexcept
        : function_base_vtable(construct_vtable<T>(), copyable_tag{}), callable_vtable<Sig>(construct_vtable<T>()) {}

    template <typename T, typename CopyableTag>
    constexpr function_vtable(construct_vtable<T>, CopyableTag) noexcept
        : function_base_vtable(construct_vtable<T>(), CopyableTag{}), callable_vtable<Sig>(construct_vtable<T>()) {}
};

template <typename Sig>
struct function_vtable<Sig, /*Copyable*/ true> : function_vtable<Sig, false> {
    using copyable_tag = std::integral_constant<bool, true>;

    template <typename T>
    constexpr function_vtable(construct_vtable<T>) noexcept
        : function_vtable<Sig, false>(construct_vtable<T>(), copyable_tag{}) {}
};

template <typename Sig>
using unique_function_vtable = function_vtable<Sig, false>;

} // namespace einsums::util::detail