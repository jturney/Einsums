//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/detail/basic_function.hpp>
#include <einsums/functional/traits/get_function_address.hpp>
#include <einsums/functional/traits/get_function_annotation.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace einsums::util::detail {
template <typename Sig>
class function;

template <typename R, typename... Ts>
class function<R(Ts...)> : public detail::basic_function<R(Ts...), true> {
    using base_type = detail::basic_function<R(Ts...), true>;

  public:
    using result_type = R;

    constexpr function(std::nullptr_t = nullptr) noexcept {}

    function(function const &)                = default;
    function(function &&) noexcept            = default;
    function &operator=(function const &)     = default;
    function &operator=(function &&) noexcept = default;

    // the split SFINAE prevents MSVC from eagerly instantiating things
    template <typename F, typename FD = std::decay_t<F>>
        requires requires {
            requires(!std::is_same_v<FD, function>);
            requires(std::is_invocable_r_v<R, FD &, Ts...>);
        }
    function(F &&f) {
        assign(std::forward<F>(f));
    }

    // the split SFINAE prevents MSVC from eagerly instantiating things
    template <typename F, typename FD = std::decay_t<F>>
        requires requires {
            requires(!std::is_same_v<FD, function>);
            requires(std::is_invocable_r_v<R, FD &, Ts...>);
        }
    function &operator=(F &&f) {
        assign(std::forward<F>(f));
        return *this;
    }

    using base_type::operator();
    using base_type::assign;
    using base_type::empty;
    using base_type::reset;
    using base_type::target;
};
} // namespace einsums::util::detail

#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
///////////////////////////////////////////////////////////////////////////////
namespace einsums::detail {
template <typename Sig>
struct get_function_address<einsums::util::detail::function<Sig>> {
    static constexpr std::size_t call(einsums::util::detail::function<Sig> const &f) noexcept { return f.get_function_address(); }
};

template <typename Sig>
struct get_function_annotation<einsums::util::detail::function<Sig>> {
    static constexpr char const *call(einsums::util::detail::function<Sig> const &f) noexcept { return f.get_function_annotation(); }
};
} // namespace einsums::detail
#endif
