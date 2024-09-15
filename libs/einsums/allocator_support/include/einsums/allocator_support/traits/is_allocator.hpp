//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <type_traits>
#include <utility>

namespace einsums::detail {
template <typename T>
struct has_allocate {
  private:
    template <typename U>
    static std::false_type test(...);

    template <typename U>
    static std::true_type test(decltype(std::declval<U>().allocate(0)));

  public:
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template <typename T>
struct has_value_type {
  private:
    template <typename U>
    static std::false_type test(...);

    template <typename U>
    static std::true_type test(typename U::value_type *);

  public:
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template <typename T, bool HasAllocate = has_allocate<T>::value>
struct has_deallocate {
  private:
    using pointer = decltype(std::declval<T>().allocate(0));

    template <typename Alloc, typename Pointer>
    static auto test(Alloc &&a, Pointer &&p) -> decltype(a.deallocate(p, 0), std::true_type());

    template <typename Alloc, typename Pointer>
    static auto test(Alloc const &a, Pointer &&p) -> std::false_type;

  public:
    static constexpr bool value = decltype(test<T>(std::declval<T>(), std::declval<pointer>()))::value;
};

template <typename T>
struct has_deallocate<T, false> {
    static constexpr bool value = false;
};

///////////////////////////////////////////////////////////////////////////
template <typename T>
struct is_allocator : std::integral_constant<bool, has_value_type<T>::value && has_allocate<T>::value && has_deallocate<T>::value> {};

template <typename T>
inline constexpr bool is_allocator_v = is_allocator<T>::value;

} // namespace einsums::detail