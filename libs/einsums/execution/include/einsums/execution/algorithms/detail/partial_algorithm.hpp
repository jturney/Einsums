//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/datastructures/member_pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace einsums::execution::experimental::detail {
template <typename Tag, typename IsPack, typename... Ts>
struct partial_algorithm_base;

template <typename Tag, std::size_t... Is, typename... Ts>
struct partial_algorithm_base<Tag, einsums::util::detail::index_pack<Is...>, Ts...> {
  private:
    einsums::util::detail::member_pack_for<std::decay_t<Ts>...> ts;

  public:
    template <typename... Ts_>
    explicit constexpr partial_algorithm_base(Ts_ &&...ts) : ts(std::piecewise_construct, std::forward<Ts_>(ts)...) {}

                            partial_algorithm_base(partial_algorithm_base &&)      = default;
    partial_algorithm_base &operator=(partial_algorithm_base &&)                   = default;
                            partial_algorithm_base(partial_algorithm_base const &) = delete;
    partial_algorithm_base &operator=(partial_algorithm_base const &)              = delete;

    template <typename U>
    friend constexpr EINSUMS_FORCEINLINE auto operator|(U &&u, partial_algorithm_base p) {
        return Tag{}(std::forward<U>(u), std::move(p.ts).template get<Is>()...);
    }
};

template <typename Tag, typename... Ts>
using partial_algorithm = partial_algorithm_base<Tag, typename einsums::util::detail::make_index_pack<sizeof...(Ts)>::type, Ts...>;
} // namespace einsums::execution::experimental::detail
