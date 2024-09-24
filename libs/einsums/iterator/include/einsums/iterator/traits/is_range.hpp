//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>
#include <einsums/iterator/range.hpp>
#include <einsums/iterator/traits/is_sentinel_for.hpp>

#include <iterator>
#include <type_traits>

namespace einsums::traits {
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct is_range : std::false_type
    {
    };

    template <typename T>
    struct is_range<T,
        typename std::enable_if<
            einsums::traits::is_sentinel_for<typename util::detail::sentinel<T>::type,
                typename util::detail::iterator<T>::type>::value>::type> : std::true_type
    {
    };

    template <typename T>
    inline constexpr bool is_range_v = is_range<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct range_iterator : util::detail::iterator<T>
    {
    };

    template <typename T>
    using range_iterator_t = typename range_iterator<T>::type;

    template <typename T, typename Enable = void>
    struct range_sentinel : util::detail::sentinel<T>
    {
    };

    template <typename T>
    using range_sentinel_t = typename range_sentinel<T>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, bool IsRange = is_range<R>::value>
    struct range_traits
    {
    };

    template <typename R>
    struct range_traits<R, true> : std::iterator_traits<typename util::detail::iterator<R>::type>
    {
        using iterator_type = typename util::detail::iterator<R>::type;
        using sentinel_type = typename util::detail::sentinel<R>::type;
    };

    template <typename T>
    using range_iterator_t = typename range_iterator<T>::type;
}    // namespace einsums::traits
