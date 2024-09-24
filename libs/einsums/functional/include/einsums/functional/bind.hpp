//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/datastructures/member_pack.hpp>
#include <einsums/functional/invoke.hpp>
#include <einsums/functional/one_shot.hpp>
#include <einsums/functional/traits/get_function_address.hpp>
#include <einsums/functional/traits/get_function_annotation.hpp>
#include <einsums/functional/traits/is_bind_expression.hpp>
#include <einsums/type_support/decay.hpp>
#include <einsums/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace einsums::util::detail {
template <std::size_t I>
struct bind_eval_placeholder {
    template <typename T, typename... Us>
    static constexpr EINSUMS_HOST_DEVICE decltype(auto) call(T && /*t*/, Us &&...vs) {
        return util::detail::member_pack_for<Us &&...>(std::piecewise_construct, std::forward<Us>(vs)...).template get<I>();
    }
};

template <typename T, std::size_t NumUs, typename TD = std::decay_t<T>, typename Enable = void>
struct bind_eval {
    template <typename... Us>
    static constexpr EINSUMS_HOST_DEVICE T &&call(T &&t, Us &&.../*vs*/) {
        return std::forward<T>(t);
    }
};

template <typename T, std::size_t NumUs, typename TD>
struct bind_eval<T, NumUs, TD, std::enable_if_t<std::is_placeholder_v<TD> != 0 && (std::is_placeholder_v<TD> <= NumUs)>>
    : bind_eval_placeholder<(std::size_t)std::is_placeholder_v<TD> - 1> {};

template <typename T, std::size_t NumUs, typename TD>
struct bind_eval<T, NumUs, TD, std::enable_if_t<einsums::detail::is_bind_expression_v<TD>>> {
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    static constexpr EINSUMS_HOST_DEVICE std::invoke_result_t<T, Us...> call(T &&t, Us &&...vs) {
        return EINSUMS_INVOKE(std::forward<T>(t), std::forward<Us>(vs)...);
    }
};

///////////////////////////////////////////////////////////////////////
template <typename F, typename Ts, typename... Us>
struct invoke_bound_result;

template <typename F, typename... Ts, typename... Us>
struct invoke_bound_result<F, util::detail::pack<Ts...>, Us...>
    : std::invoke_result<F, decltype(bind_eval<Ts, sizeof...(Us)>::call(std::declval<Ts>(), std::declval<Us>()...))...> {};

template <typename F, typename Ts, typename... Us>
using invoke_bound_result_t = typename invoke_bound_result<F, Ts, Us...>::type;

///////////////////////////////////////////////////////////////////////
template <typename F, typename Is, typename... Ts>
class bound;

template <typename F, std::size_t... Is, typename... Ts>
class bound<F, index_pack<Is...>, Ts...> {
  public:
    template <typename F_, typename... Ts_, typename = std::enable_if_t<std::is_constructible_v<F, F_>>>
    constexpr explicit bound(F_ &&f, Ts_ &&...vs) : _f(std::forward<F_>(f)), _args(std::piecewise_construct, std::forward<Ts_>(vs)...) {}

#if !defined(__NVCC__) && !defined(__CUDACC__)
    bound(bound const &) = default;
    bound(bound &&)      = default;
#else
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    constexpr EINSUMS_HOST_DEVICE bound(bound const &other) : _f(other._f), _args(other._args) {}

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    constexpr EINSUMS_HOST_DEVICE bound(bound &&other) : _f(std::move(other._f)), _args(std::move(other._args)) {}
#endif

    bound &operator=(bound const &) = delete;

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    constexpr EINSUMS_NO_SANITIZE_ADDRESS EINSUMS_HOST_DEVICE invoke_bound_result_t<F &, util::detail::pack<Ts &...>, Us &&...>
                                                              operator()(Us &&...vs)                                                           &{
        return EINSUMS_INVOKE(_f, detail::bind_eval<Ts &, sizeof...(Us)>::call(_args.template get<Is>(), std::forward<Us>(vs)...)...);
    }

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    constexpr EINSUMS_HOST_DEVICE invoke_bound_result_t<F const &, util::detail::pack<Ts const &...>, Us &&...>
                                  operator()(Us &&...vs) const                               &{
        return EINSUMS_INVOKE(_f, detail::bind_eval<Ts const &, sizeof...(Us)>::call(_args.template get<Is>(), std::forward<Us>(vs)...)...);
    }

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    constexpr EINSUMS_HOST_DEVICE invoke_bound_result_t<F &&, util::detail::pack<Ts &&...>, Us &&...> operator()(Us &&...vs) && {
        return EINSUMS_INVOKE(std::move(_f),
                              detail::bind_eval<Ts, sizeof...(Us)>::call(std::move(_args).template get<Is>(), std::forward<Us>(vs)...)...);
    }

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Us>
    constexpr EINSUMS_HOST_DEVICE invoke_bound_result_t<F const &&, util::detail::pack<Ts const &&...>, Us &&...>
                                  operator()(Us &&...vs) const                               &&{
        return EINSUMS_INVOKE(std::move(_f), detail::bind_eval<Ts const, sizeof...(Us)>::call(std::move(_args).template get<Is>(),
                                                                                                                            std::forward<Us>(vs)...)...);
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

///////////////////////////////////////////////////////////////////////////
template <typename F, typename... Ts>
constexpr bound<std::decay_t<F>, util::detail::make_index_pack_t<sizeof...(Ts)>, ::einsums::detail::decay_unwrap_t<Ts>...>
bind(F &&f, Ts &&...vs) {
    using result_type = bound<std::decay_t<F>, util::detail::make_index_pack_t<sizeof...(Ts)>, ::einsums::detail::decay_unwrap_t<Ts>...>;

    return result_type(std::forward<F>(f), std::forward<Ts>(vs)...);
}
} // namespace einsums::util::detail

namespace einsums::detail {
template <typename F, typename... Ts>
struct is_bind_expression<einsums::util::detail::bound<F, Ts...>> : std::true_type {};

///////////////////////////////////////////////////////////////////////////
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
template <typename F, typename... Ts>
struct get_function_address<einsums::util::detail::bound<F, Ts...>> {
    static constexpr std::size_t call(einsums::util::detail::bound<F, Ts...> const &f) noexcept { return f.get_function_address(); }
};

///////////////////////////////////////////////////////////////////////////
template <typename F, typename... Ts>
struct get_function_annotation<einsums::util::detail::bound<F, Ts...>> {
    static constexpr char const *call(einsums::util::detail::bound<F, Ts...> const &f) noexcept { return f.get_function_annotation(); }
};

#    if EINSUMS_HAVE_ITTNOTIFY != 0 && !defined(EINSUMS_HAVE_APEX)
template <typename F, typename... Ts>
struct get_function_annotation_itt<einsums::util::detail::bound<F, Ts...>> {
    static util::itt::string_handle call(einsums::util::detail::bound<F, Ts...> const &f) noexcept {
        return f.get_function_annotation_itt();
    }
};
#    endif
#endif
} // namespace einsums::detail
