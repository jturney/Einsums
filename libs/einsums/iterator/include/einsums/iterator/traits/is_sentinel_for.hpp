//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/iterator/traits/is_iterator.hpp>

#include <type_traits>
#include <utility>

namespace einsums::traits {

    ///////////////////////////////////////////////////////////////////////////
    // The trait checks whether sentinel Sent is proper for iterator I.
    // There are two requirements for this:
    // 1. iterator I should be an input or output iterator
    // 2. I and S should oblige with the weakly-equality-comparable concept
    template <typename Sent, typename Iter, typename Enable = void>
    struct is_sentinel_for : std::false_type
    {
    };

    template <typename Sent, typename Iter>
    struct is_sentinel_for<Sent, Iter,
        typename std::enable_if_t<
            is_iterator_v<Iter>&& ::einsums::detail::is_weakly_equality_comparable_with_v<Iter, Sent>>>
      : std::true_type
    {
    };

    template <typename Sent, typename Iter>
    inline constexpr bool is_sentinel_for_v = is_sentinel_for<Sent, Iter>::value;

    ///////////////////////////////////////////////////////////////////////////
#if defined(EINSUMS_HAVE_CXX20_STD_DISABLE_SIZED_SENTINEL_FOR)
    template <typename Sent, typename Iter>
    inline constexpr bool disable_sized_sentinel_for = std::disable_sized_sentinel_for<Sent, Iter>;
#else
    template <typename Sent, typename Iter>
    inline constexpr bool disable_sized_sentinel_for = false;
#endif

    template <typename Sent, typename Iter, typename Enable = void>
    struct is_sized_sentinel_for : std::false_type
    {
    };

    template <typename Sent, typename Iter>
    struct is_sized_sentinel_for<Sent, Iter,
        std::void_t<std::enable_if_t<einsums::traits::is_sentinel_for<Sent, Iter>::value &&
                        !disable_sized_sentinel_for<typename std::remove_cv<Sent>::type,
                            typename std::remove_cv<Iter>::type>>,
            typename detail::subtraction_result<Iter, Sent>::type,
            typename detail::subtraction_result<Sent, Iter>::type>> : std::true_type
    {
    };

    template <typename Sent, typename Iter>
    inline constexpr bool is_sized_sentinel_for_v = is_sized_sentinel_for<Sent, Iter>::value;

}    // namespace einsums::traits
