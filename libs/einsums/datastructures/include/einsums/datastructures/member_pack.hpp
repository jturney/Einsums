//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/type_support/pack.hpp>

#include <cstddef> // for size_t
#include <type_traits>
#include <utility>

namespace einsums::util::detail {
#if defined(EINSUMS_HAVE_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE) && !(defined(EINSUMS_CUDA_VERSION) && EINSUMS_CUDA_VERSION >= 1200)
template <std::size_t I, typename T>
struct member_leaf {
    EINSUMS_NO_UNIQUE_ADDRESS T member;

    member_leaf() = default;

    template <typename U>
    explicit constexpr member_leaf(std::piecewise_construct_t, U &&v) : member(std::forward<U>(v)) {}
};

template <std::size_t I, typename T>
T member_type(member_leaf<I, T> const & /*leaf*/) noexcept;

template <std::size_t I, typename T>
static constexpr T &member_get(member_leaf<I, T> &leaf) noexcept {
    return leaf.member;
}
template <std::size_t I, typename T>
static constexpr T const &member_get(member_leaf<I, T> const &leaf) noexcept {
    return leaf.member;
}
#else
template <std::size_t I, typename T, bool Empty = std::is_empty<T>::value && !std::is_final<T>::value>
struct member_leaf {
    T member;

    member_leaf() = default;

    template <typename U>
    explicit constexpr member_leaf(std::piecewise_construct_t, U &&v) : member(std::forward<U>v)) {}
};

template <std::size_t I, typename T>
struct member_leaf<I, T, /*Empty*/ true> : T {
    member_leaf() = default;

    template <typename U>
    explicit constexpr member_leaf(std::piecewise_construct_t, U &&v) : T(std::forward<U>v)) {}
};

template <std::size_t I, typename T>
T member_type(member_leaf<I, T> const & /*leaf*/) noexcept;

template <std::size_t I, typename T>
static constexpr T &member_get(member_leaf<I, T, false> &leaf) noexcept {
    return leaf.member;
}
template <std::size_t I, typename T>
static constexpr T &member_get(member_leaf<I, T, true> &leaf) noexcept {
    return leaf;
}
template <std::size_t I, typename T>
static constexpr T const &member_get(member_leaf<I, T, false> const &leaf) noexcept {
    return leaf.member;
}
template <std::size_t I, typename T>
static constexpr T const &member_get(member_leaf<I, T, true> const &leaf) noexcept {
    return leaf;
}
#endif

///////////////////////////////////////////////////////////////////////
template <typename Is, typename... Ts>
struct EINSUMS_EMPTY_BASES member_pack;

template <std::size_t... Is, typename... Ts>
struct EINSUMS_EMPTY_BASES member_pack<util::detail::index_pack<Is...>, Ts...> : member_leaf<Is, Ts>... {
    member_pack() = default;

    template <typename... Us>
    explicit constexpr member_pack(std::piecewise_construct_t, Us &&...us)
        : member_leaf<Is, Ts>(std::piecewise_construct, std::forward<Us>(us))... {}

    template <std::size_t I>
    constexpr decltype(auto) get() & noexcept {
        return member_get<I>(*this);
    }
    template <std::size_t I>
    constexpr decltype(auto) get() const & noexcept {
        return member_get<I>(*this);
    }
    template <std::size_t I>
    constexpr decltype(auto) get() && noexcept {
        using T = decltype(member_type<I>(*this));
        return static_cast<T &&>(member_get<I>(*this));
    }
    template <std::size_t I>
    constexpr decltype(auto) get() const && noexcept {
        using T = decltype(member_type<I>(*this));
        return static_cast<T &&>(member_get<I>(*this));
    }
};

template <typename... Ts>
using member_pack_for = member_pack<util::detail::make_index_pack_t<sizeof...(Ts)>, Ts...>;

} // namespace einsums::util::detail
