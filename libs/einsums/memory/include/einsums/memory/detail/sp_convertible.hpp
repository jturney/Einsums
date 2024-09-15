//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstddef>

namespace einsums::memory::detail {

template <typename Y, typename T>
struct sp_convertible {
    using yes = char (&)[1];
    using no  = char (&)[2];

    static yes f(T *);
    static no  f(...);

    static constexpr bool value = sizeof((f)(static_cast<Y *>(nullptr))) == sizeof(yes);
};

template <typename Y, typename T>
struct sp_convertible<Y, T[]> {
    static constexpr bool value = false;
};

template <typename Y, typename T>
struct sp_convertible<Y[], T[]> {
    static constexpr bool value = sp_convertible<Y[1], T[1]>::value;
};

template <typename Y, std::size_t N, typename T>
struct sp_convertible<Y[N], T[]> {
    static constexpr bool value = sp_convertible<Y[1], T[1]>::value;
};

template <typename Y, typename T>
inline constexpr bool sp_convertible_v = sp_convertible<Y, T>::value;

} // namespace einsums::memory::detail