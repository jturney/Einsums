//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/invoke.hpp>
#include <einsums/functional/one_shot.hpp>
#include <einsums/functional/traits/get_function_address.hpp>
#include <einsums/functional/traits/get_function_annotation.hpp>
#include <einsums/type_support/decay.hpp>
#include <einsums/datastructures/member_pack.hpp>
#include <einsums/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace einsums::util::detail {
template <typename F, typename Ts, typename... Us>
struct invoke_bound_back_result;

template <typename F, typename... Ts, typename... Us>
struct invoke_bound_back_result<F, util::detail::pack<Ts...>, Us...> : std::invoke_result<F, Us..., Ts...> {};

///////////////////////////////////////////////////////////////////////
template <typename F, typename Is, typename... Ts>
class bound_back;

template <typename F, std::size_t... Is, typename... Ts>
class bound_back<F, index_pack<Is...>, Ts...> {
  public:
    template <typename F_, typename... Ts_, typename = std::enable_if_t<std::is_constructible_v<F, F_>>>
    constexpr explicit bound_back(F_ &&f, Ts_ &&...vs)
        : _f(std::forward<F_>(f)), _args(std::piecewise_construct, std::forward<Ts_>(vs)...) {}

#if !defined(__NVCC__) && !defined(__CUDACC__)
    bound_back(bound_back const &) = default;
    bound_back(bound_back &&)      = default;
#else
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    constexpr EINSUMS_HOST_DEVICE bound_back(bound_back const &other) : _f(other._f), _args(other._args) {}

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    constexpr EINSUMS_HOST_DEVICE bound_back(bound_back &&other) : _f(std::move(other._f)), _args(std::move(other._args)) {}
#endif

    bound_back &operator=(bound_back const &) = delete;

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    constexpr EINSUMS_HOST_DEVICE typename invoke_bound_back_result<F &, util::detail::pack<Ts &...>, Us &&...>::type
    operator()(Us &&...vs) & {
        return EINSUMS_INVOKE(_f, std::forward<Us>(vs)..., _args.template get<Is>()...);
    }

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    constexpr EINSUMS_HOST_DEVICE typename invoke_bound_back_result<F const &, util::detail::pack<Ts const &...>, Us &&...>::type
    operator()(Us &&...vs) const & {
        return EINSUMS_INVOKE(_f, std::forward<Us>(vs)..., _args.template get<Is>()...);
    }

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    constexpr EINSUMS_HOST_DEVICE typename invoke_bound_back_result<F &&, util::detail::pack<Ts &&...>, Us &&...>::type
    operator()(Us &&...vs) && {
        return EINSUMS_INVOKE(std::move(_f), std::forward<Us>(vs)..., std::move(_args).template get<Is>()...);
    }

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    constexpr EINSUMS_HOST_DEVICE typename invoke_bound_back_result<F const &&, util::detail::pack<Ts const &&...>, Us &&...>::type
    operator()(Us &&...vs) const && {
        return EINSUMS_INVOKE(std::move(_f), std::forward<Us>(vs)..., std::move(_args).template get<Is>()...);
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
constexpr bound_back<std::decay_t<F>, util::detail::make_index_pack_t<sizeof...(Ts)>, ::einsums::detail::decay_unwrap_t<Ts>...>
bind_back(F &&f, Ts &&...vs) {
    using result_type =
        bound_back<std::decay_t<F>, util::detail::make_index_pack_t<sizeof...(Ts)>, ::einsums::detail::decay_unwrap_t<Ts>...>;

    return result_type(std::forward<F>(f), std::forward<Ts>(vs)...);
}

// nullary functions do not need to be bound again
template <typename F>
constexpr std::decay_t<F> bind_back(F &&f) {
    return std::forward<F>(f);
}
} // namespace einsums::util::detail

namespace einsums::detail {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
template <typename F, typename... Ts>
struct get_function_address<einsums::util::detail::bound_back<F, Ts...>> {
    static constexpr std::size_t call(einsums::util::detail::bound_back<F, Ts...> const &f) noexcept { return f.get_function_address(); }
};

///////////////////////////////////////////////////////////////////////////
template <typename F, typename... Ts>
struct get_function_annotation<einsums::util::detail::bound_back<F, Ts...>> {
    static constexpr char const *call(einsums::util::detail::bound_back<F, Ts...> const &f) noexcept { return f.get_function_annotation(); }
};
#endif
} // namespace einsums::detail
