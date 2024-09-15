//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/invoke.hpp>
#include <einsums/functional/one_shot.hpp>
#include <einsums/functional/traits/get_function_address.hpp>
#include <einsums/functional/traits/get_function_annotation.hpp>
#include <einsums/type_support/decay.hpp>
#include <einsums/type_support/member_pack.hpp>
#include <einsums/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace einsums::util::detail {
template <typename F, typename Ts, typename... Us>
struct invoke_bound_front_result;

template <typename F, typename... Ts, typename... Us>
struct invoke_bound_front_result<F, util::detail::pack<Ts...>, Us...> : std::invoke_result<F, Ts..., Us...> {};

///////////////////////////////////////////////////////////////////////
template <typename F, typename Is, typename... Ts>
class bound_front;

template <typename F, std::size_t... Is, typename... Ts>
class bound_front<F, index_pack<Is...>, Ts...> {
  public:
    template <typename F_, typename... Ts_, typename = typename std::enable_if_t<std::is_constructible_v<F, F_>>>
    constexpr explicit bound_front(F_ &&f, Ts_ &&...vs)
        : _f(std::forward<F_>(f)), _args(std::piecewise_construct, std::forward<Ts_>(vs)...) {}

#if !defined(__NVCC__) && !defined(__CUDACC__)
    bound_front(bound_front const &) = default;
    bound_front(bound_front &&)      = default;
#else
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    constexpr EINSUMS_HOST_DEVICE bound_front(bound_front const &other) : _f(other._f), _args(other._args) {}

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    constexpr EINSUMS_HOST_DEVICE bound_front(bound_front &&other) : _f(std::move(other._f)), _args(std::move(other._args)) {}
#endif

    bound_front &operator=(bound_front const &) = delete;

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    constexpr EINSUMS_HOST_DEVICE typename invoke_bound_front_result<F &, util::detail::pack<Ts &...>, Us &&...>::type
    operator()(Us &&...vs) & {
        return EINSUMS_INVOKE(_f, _args.template get<Is>()..., std::forward<Us>(vs)...);
    }

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    constexpr EINSUMS_HOST_DEVICE typename invoke_bound_front_result<F const &, util::detail::pack<Ts const &...>, Us &&...>::type
    operator()(Us &&...vs) const & {
        return EINSUMS_INVOKE(_f, _args.template get<Is>()..., std::forward<Us>(vs)...);
    }

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    constexpr EINSUMS_HOST_DEVICE typename invoke_bound_front_result<F &&, util::detail::pack<Ts &&...>, Us &&...>::type
    operator()(Us &&...vs) && {
        return EINSUMS_INVOKE(std::move(_f), std::move(_args).template get<Is>()..., std::forward<Us>(vs)...);
    }

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    constexpr EINSUMS_HOST_DEVICE typename invoke_bound_front_result<F const &&, util::detail::pack<Ts const &&...>, Us &&...>::type
    operator()(Us &&...vs) const && {
        return EINSUMS_INVOKE(std::move(_f), std::move(_args).template get<Is>()..., std::forward<Us>(vs)...);
    }

    constexpr std::size_t get_function_address() const { return einsums::detail::get_function_address<F>::call(_f); }

    constexpr char const *get_function_annotation() const {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
        return einsums::detail::get_function_annotation<F>::call(_f);
#else
        return nullptr;
#endif
    }

  private:
    F                                    _f;
    util::detail::member_pack_for<Ts...> _args;
};

template <typename F, typename... Ts>
constexpr bound_front<std::decay_t<F>, typename util::detail::make_index_pack<sizeof...(Ts)>::type, std::decay_t<Ts>...>
bind_front(F &&f, Ts &&...vs) {
    using result_type = bound_front<std::decay_t<F>, typename util::detail::make_index_pack<sizeof...(Ts)>::type, std::decay_t<Ts>...>;

    return result_type(std::forward<F>(f), std::forward<Ts>(vs)...);
}

// nullary functions do not need to be bound again
template <typename F>
constexpr std::decay_t<F> bind_front(F &&f) {
    return std::forward<F>(f);
}
} // namespace einsums::util::detail

namespace einsums::detail {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
template <typename F, typename... Ts>
struct get_function_address<einsums::util::detail::bound_front<F, Ts...>> {
    static constexpr std::size_t call(einsums::util::detail::bound_front<F, Ts...> const &f) noexcept { return f.get_function_address(); }
};

///////////////////////////////////////////////////////////////////////////
template <typename F, typename... Ts>
struct get_function_annotation<einsums::util::detail::bound_front<F, Ts...>> {
    static constexpr char const *call(einsums::util::detail::bound_front<F, Ts...> const &f) noexcept {
        return f.get_function_annotation();
    }
};
#endif
} // namespace einsums::detail
