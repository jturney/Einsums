//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/functional/invoke.hpp>
#include <einsums/functional/traits/get_function_address.hpp>
#include <einsums/functional/traits/get_function_annotation.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace einsums::util::detail {
template <typename F>
class one_shot_wrapper {
  public:
    template <typename F_, typename = std::enable_if_t<std::is_constructible_v<F, F_>>>
    constexpr explicit one_shot_wrapper(F_ &&f)
        : _f(std::forward<F>(f))
#if defined(EINSUMS_DEBUG)
          ,
          _called(false)
#endif
    {
    }

    constexpr one_shot_wrapper(one_shot_wrapper &&other)
        : _f(std::move(other._f))
#if defined(EINSUMS_DEBUG)
          ,
          _called(other._called)
#endif
    {
#if defined(EINSUMS_DEBUG)
        other._called = true;
#endif
    }

    void check_call() {
#if defined(EINSUMS_DEBUG)
        EINSUMS_ASSERT(!_called);
        _called = true;
#endif
    }

    template <typename... Ts>
    constexpr EINSUMS_HOST_DEVICE std::invoke_result_t<F, Ts...> operator()(Ts &&...vs) {
        check_call();

        return EINSUMS_INVOKE(std::move(_f), std::forward<Ts>(vs)...);
    }

    constexpr std::size_t get_function_address() const { return einsums::detail::get_function_address<F>::call(_f); }

    constexpr char const *get_function_annotation() const {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
        return einsums::detail::get_function_annotation<F>::call(_f);
#else
        return nullptr;
#endif
    }

  public: // exposition-only
    F _f;
#if defined(EINSUMS_DEBUG)
    bool _called;
#endif
};

template <typename F>
constexpr one_shot_wrapper<std::decay_t<F>> one_shot(F &&f) {
    using result_type = one_shot_wrapper<std::decay_t<F>>;

    return result_type(std::forward<F>(f));
}
} // namespace einsums::util::detail

namespace einsums::detail {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
template <typename F>
struct get_function_address<util::detail::one_shot_wrapper<F>> {
    static constexpr std::size_t call(util::detail::one_shot_wrapper<F> const &f) noexcept { return f.get_function_address(); }
};

///////////////////////////////////////////////////////////////////////////
template <typename F>
struct get_function_annotation<util::detail::one_shot_wrapper<F>> {
    static constexpr char const *call(util::detail::one_shot_wrapper<F> const &f) noexcept { return f.get_function_annotation(); }
};

#endif
} // namespace einsums::detail
