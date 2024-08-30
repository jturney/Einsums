//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <memory>
#include <type_traits>

namespace einsums {

namespace detail {
template <typename T>
struct smart_pointer_helper : std::false_type {};

template <typename T>
struct smart_pointer_helper<std::shared_ptr<T>> : std::true_type {};

template <typename T>
struct smart_pointer_helper<std::unique_ptr<T>> : std::true_type {};

template <typename T>
struct smart_pointer_helper<std::weak_ptr<T>> : std::true_type {};

template <typename T>
struct is_smart_pointer_helper : smart_pointer_helper<std::remove_cvref_t<T>> {};
} // namespace detail

template <typename T>
inline constexpr bool is_smart_pointer_v = detail::is_smart_pointer_helper<T>::value;

template <typename T>
concept SmartPointer = is_smart_pointer_v<T>;

template <typename T>
concept NonSmartPointer = !is_smart_pointer_v<T>;

} // namespace einsums