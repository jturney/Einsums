//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <type_traits>
#include <utility>

namespace einsums {

template <typename T>
struct IsIntegralOrEnum {
  private:
    using UnderlyingType = std::remove_reference_t<T>;

  public:
    static bool const value = !std::is_class_v<UnderlyingType> && // Filter conversion operators.
                              !std::is_pointer_v<UnderlyingType> && !std::is_floating_point_v<UnderlyingType> &&
                              (std::is_enum_v<UnderlyingType> || std::is_convertible_v<UnderlyingType, unsigned long long>);
};

/// If T is a pointer, just turn it. If it is not, return T&
template <typename T>
struct AddLValueReferenceIfNotPointer {
    using type = std::conditional_t<std::is_pointer_v<T>, T, T &>;
};

/// If T is a pointer to X, return a pointer to const X. If it is not,
/// return const T.
template <typename T>
struct AddConstPastPointer {
    using type = std::conditional_t<std::is_pointer_v<T>, std::remove_pointer_t<T> const *, T const>;
};

template <typename T>
struct ConstPointerOrConstRef {
    using type = std::conditional_t<std::is_pointer_v<T>, typename AddConstPastPointer<T>::type, T const &>;
};

} // namespace einsums