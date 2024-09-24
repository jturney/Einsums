//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>

namespace einsums::execution::experimental {
template <typename OperationState>
inline constexpr bool is_operation_state_v = operation_state<OperationState>;
}
#else

#    include <einsums/functional/tag_invoke.hpp>

#    include <type_traits>
#    include <utility>

namespace einsums::execution::experimental {
template <typename O>
struct is_operation_state;

inline constexpr struct start_t : einsums::functional::detail::tag_noexcept<start_t> {
} start{};

namespace detail {
template <bool IsOperationState, typename O>
struct is_operation_state_impl;

template <typename O>
struct is_operation_state_impl<false, O> : std::false_type {};

template <typename O>
struct is_operation_state_impl<true, O> : std::integral_constant<bool, noexcept(start(std::declval<O &>()))> {};
} // namespace detail

template <typename O>
struct is_operation_state
    : detail::is_operation_state_impl<std::is_destructible_v<O> && std::is_object_v<O> && std::is_invocable_v<start_t, std::decay_t<O> &>,
                                      O> {};

template <typename O>
inline constexpr bool is_operation_state_v = is_operation_state<O>::value;
} // namespace einsums::execution::experimental
#endif