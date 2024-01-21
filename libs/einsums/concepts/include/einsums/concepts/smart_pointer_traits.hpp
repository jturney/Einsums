//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <memory>
#include <type_traits>

namespace einsums {

namespace details {
template <typename T>
struct IsSmartPointerHelper : public std::false_type {};

template <typename T>
struct IsSmartPointerHelper<std::shared_ptr<T>> : public std::true_type {};

template <typename T>
struct IsSmartPointerHelper<std::unique_ptr<T>> : public std::true_type {};

template <typename T>
struct IsSmartPointerHelper<std::weak_ptr<T>> : public std::true_type {};
} // namespace details

template <typename T>
struct IsSmartPointerHelper : public details::IsSmartPointerHelper<typename std::remove_cv<T>::type> {};

template <typename T>
inline constexpr bool IsSmartPointerV = IsSmartPointerHelper<T>::value;

template <typename T>
concept SmartPointer = IsSmartPointerV<T>;

template <typename T>
concept NotASmartPointer = !
IsSmartPointerV<T>;

} // namespace einsums