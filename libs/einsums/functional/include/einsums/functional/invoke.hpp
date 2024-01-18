//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/detail/invoke.hpp>
#include <einsums/type_support/void_guard.hpp>

#include <type_traits>
#include <utility>

namespace einsums::util::detail {

#define EINSUMS_INVOKE_R(R, F, ...) (::einsums::detail::void_guard<R>(), EINSUMS_INVOKE(F, __VA_ARGS__))

/// Invokes the given callable object f with the content of
/// the argument pack vs
///
/// \param f Requires to be a callable object.
///          If f is a member function pointer, the first argument in
///          the pack will be treated as the callee (this object).
///
/// \param vs An arbitrary pack of arguments
///
/// \returns The result of the callable object when it's called with
///          the given argument types.
///
/// \throws std::exception like objects thrown by call to object f
///         with the argument types vs.
///
/// \note This function is similar to `std::invoke` (C++17)
template <typename F, typename... Ts>
constexpr EINSUMS_HOST_DEVICE auto invoke(F &&f, Ts &&...vs) -> std::invoke_result_t<F, Ts...> {
    return EINSUMS_INVOKE(EINSUMS_FORWARD(F, f), EINSUMS_FORWARD(Ts, vs)...);
}

///////////////////////////////////////////////////////////////////////////
/// \copydoc invoke
///
/// \tparam R The result type of the function when it's called
///           with the content of the given argument types vs.
template <typename R, typename F, typename... Ts>
constexpr EINSUMS_HOST_DEVICE auto invoke_r(F &&f, Ts &&...vs) -> R {
    return EINSUMS_INVOKE_R(R, EINSUMS_FORWARD(F, f), EINSUMS_FORWARD(Ts, vs)...);
}

} // namespace einsums::util::detail