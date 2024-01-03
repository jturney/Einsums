//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <tuple>

namespace einsums::arguments {

namespace detail {

// declaration
template <class SearchPattern, int Position, int Count, bool Branch, class PrevHead, class Arguments>
struct TuplePosition;
// initialization case
template <class S, int P, int C, bool B, class not_used, class... Tail>
struct TuplePosition<S, P, C, B, not_used, std::tuple<Tail...>> : TuplePosition<S, P, C, false, not_used, std::tuple<Tail...>> {};
// recursive case
template <class S, int P, int C, class not_used, class Head, class... Tail>
struct TuplePosition<S, P, C, false, not_used, std::tuple<Head, Tail...>>
    : TuplePosition<S, P + 1, C, std::is_convertible_v<Head, S>, Head, std::tuple<Tail...>> {};
// match case
template <class S, int P, int C, class Type, class... Tail>
struct TuplePosition<S, P, C, true, Type, std::tuple<Tail...>> : std::integral_constant<int, P> {
    using type                    = Type;
    static constexpr bool present = true;
};
// default case
template <class S, class H, int P, int C>
struct TuplePosition<S, P, C, false, H, std::tuple<>> : std::integral_constant<int, -1> {
    static constexpr bool present = false;
};

} // namespace detail

template <typename SearchPattern, typename... Args>
struct TuplePosition : detail::TuplePosition<SearchPattern const &, -1, 0, false, void, std::tuple<Args...>> {};

template <typename SearchPattern, typename... Args,
          typename Idx = TuplePosition<SearchPattern const &, Args const &..., SearchPattern const &>>
auto get(SearchPattern const &definition, Args &&...args) -> typename Idx::type & {
    auto tuple = std::forward_as_tuple(args..., definition);
    return std::get<Idx::value>(tuple);
}

template <typename SearchPattern, typename... Args>
auto get(Args &&...args) -> SearchPattern & {
    auto tuple = std::forward_as_tuple(args...);
    return std::get<SearchPattern>(tuple);
}

template <int Idx, typename... Args>
auto getn(Args &&...args) -> typename std::tuple_element<Idx, std::tuple<Args...>>::type & {
    auto tuple = std::forward_as_tuple(args...);
    return std::get<Idx>(tuple);
}

// Does the parameter pack contain at least one of Type
template <typename T, typename... List>
struct Contains : std::true_type {};

template <typename T, typename Head, typename... Remaining>
struct Contains<T, Head, Remaining...> : std::conditional_t<std::is_same_v<T, Head>, std::true_type, Contains<T, Remaining...>> {};

template <typename T>
struct Contains<T> : std::false_type {};

} // namespace einsums::arguments