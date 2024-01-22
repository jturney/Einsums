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
struct unique_function;

template <typename R, typename... Ts>
struct unique_function<R(Ts...)> : public detail::basic_function<R(Ts...), false> {
  private:
    using base_type = detail::basic_function<R(Ts...), false>;

  public:
    using result_type = R;

    constexpr unique_function(std::nullptr_t = nullptr) noexcept {}

    unique_function(unique_function &&) noexcept                     = default;
    auto operator=(unique_function &&) noexcept -> unique_function & = default;

    // the split SFINAE prevents MSVC from eagerly instantiating things
    template <typename F, typename FD = std::decay_t<F>>
    unique_function(F &&f)
        requires requires {
            requires !std::is_same_v<FD, unique_function>;
            requires std::is_invocable_r_v<R, FD &, Ts...>;
        }
    {
        assign(EINSUMS_FORWARD(F, f));
    }

    // the split SFINAE prevents MSVC from eagerly instantiating things
    template <typename F, typename FD = std::decay_t<F>>
    auto operator=(F &&f) -> unique_function &
        requires requires {
            requires !std::is_same_v<FD, unique_function>;
            requires std::is_invocable_r_v<R, FD &, Ts...>;
        }
    {
        assign(EINSUMS_FORWARD(F, f));
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
namespace einsums::detail {
template <typename Sig>
struct get_function_address<util::detail::unique_function<Sig>> {
    static constexpr auto call(util::detail::unique_function<Sig> const &f) noexcept -> std::size_t {
        return f.get_function_address();
    }
};

template <typename Sig>
struct get_function_annotation<util::detail::unique_function<Sig>> {
    static constexpr auto call(util::detail::unique_function<Sig> const &f) noexcept -> char const * {
        return f.get_function_annotation();
    }
};

#    if EINSUMS_HAVE_ITTNOTIFY != 0 && !defined(EINSUMS_HAVE_APEX)
template <typename Sig>
struct get_function_annotation_itt<util::detail::unique_function<Sig>> {
    static auto call(util::detail::unique_function<Sig> const &f) noexcept -> util::itt::string_handle {
        return f.get_function_annotation_itt();
    }
};
#    endif
} // namespace einsums::detail

#endif
