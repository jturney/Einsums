//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <tuple>

namespace einsums::util::arguments {

namespace detail {

// declaration
template <class SearchPattern, int Position, int Count, bool Branch, class PrevHead, class Arguments>
struct tuple_position;
// initialization case
template <class S, int P, int C, bool B, class not_used, class... Tail>
struct tuple_position<S, P, C, B, not_used, std::tuple<Tail...>> : tuple_position<S, P, C, false, not_used, std::tuple<Tail...>> {};
// recursive case
template <class S, int P, int C, class not_used, class Head, class... Tail>
struct tuple_position<S, P, C, false, not_used, std::tuple<Head, Tail...>>
    : tuple_position<S, P + 1, C, std::is_convertible_v<Head, S>, Head, std::tuple<Tail...>> {};
// match case
template <class S, int P, int C, class Type, class... Tail>
struct tuple_position<S, P, C, true, Type, std::tuple<Tail...>> : std::integral_constant<int, P> {
    using type                    = Type;
    static constexpr bool present = true;
};
// default case
template <class S, class H, int P, int C>
struct tuple_position<S, P, C, false, H, std::tuple<>> : std::integral_constant<int, -1> {
    static constexpr bool present = false;
};

} // namespace detail

template <typename SearchPattern, typename... Args>
struct tuple_position : detail::tuple_position<const SearchPattern &, -1, 0, false, void, std::tuple<Args...>> {};

template <typename SearchPattern, typename... Args,
          typename Idx = tuple_position<const SearchPattern &, const Args &..., const SearchPattern &>>
auto get(const SearchPattern &definition, Args &&...args) -> typename Idx::type & {
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

namespace detail {

template <typename T, int Position>
constexpr auto positions_of_type() {
    return std::make_tuple();
}

template <typename T, int Position, typename Head, typename... Args>
constexpr auto positions_of_type() {
    if constexpr (std::is_convertible_v<Head, T>) {
        return std::tuple_cat(std::make_tuple(Position), positions_of_type<T, Position + 1, Args...>());
    } else {
        return positions_of_type<T, Position + 1, Args...>();
    }
}

} // namespace detail

template <typename T, typename... Args>
constexpr auto positions_of_type() {
    return detail::positions_of_type<T, 0, Args...>();
}

template <typename T, typename... Args>
constexpr auto count_of_type(/*Args... args*/) {
    return (std::is_convertible_v<Args, T> + ... + 0);
}

template <typename Result, typename Tuple>
constexpr auto get_array_from_tuple(Tuple &&tuple) -> Result {
    constexpr auto get_array = [](auto &&...x) { return Result{std::forward<decltype(x)>(x)...}; };
    return std::apply(get_array, std::forward<Tuple>(tuple));
}

template <class Tuple, class F, std::size_t... I>
constexpr auto for_each_impl(Tuple &&t, F &&f, std::index_sequence<I...>) -> F {
    return (void)std::initializer_list<int>{(std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))), 0)...}, f;
}

template <class Tuple, class F>
constexpr auto for_each(Tuple &&t, F &&f) -> F {
    return for_each_impl(std::forward<Tuple>(t), std::forward<F>(f),
                         std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template <typename ReturnType, typename Tuple>
inline auto get_from_tuple(Tuple &&tuple, std::size_t index) -> ReturnType {
    std::size_t currentIndex = 0;
    ReturnType  returnValue{-1ul};

    for_each(tuple, [index, &currentIndex, &returnValue](auto &&value) {
        if (currentIndex == index) {
            // action(std::forward<decltype(value)>(value));
            if constexpr (std::is_convertible_v<ReturnType, std::remove_reference_t<decltype(value)>>)
                returnValue = value;
        }
        ++currentIndex;
    });
    return returnValue;
}

} // namespace einsums::util::arguments