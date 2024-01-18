//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <functional>
#include <type_traits>

namespace einsums::util::detail {

template <typename C, typename T>
static constexpr auto mem_ptr_target(T &&v) noexcept -> T &&
    requires std::is_base_of_v<C, std::remove_reference_t<T>>
{
    return EINSUMS_FORWARD(T, v);
}

template <typename C, typename T>
static constexpr auto mem_ptr_target(std::reference_wrapper<T> v) noexcept -> T & {
    return v.get();
}

template <typename C, typename T>
static constexpr auto mem_ptr_target(T &&v) noexcept(
#if defined(EINSUMS_CUDA_VERSION)
    noexcept(*std::forward<T>(v))) -> decltype(*std::forward<T>(v))
#else
    noexcept(*EINSUMS_FORWARD(T, v))) -> decltype(*EINSUMS_FORWARD(T, v))
#endif
{
#if defined(EINSUMS_CUDA_VERSION)
    return *std::forward<T>(v);
#else
    return *EINSUMS_FORWARD(T, v);
#endif
}

template <typename T, typename C>
struct invoke_mem_obj {
    T C::*pm;

  public:
    constexpr invoke_mem_obj(T C::*pm) noexcept : pm(pm) {}

    template <typename T1>
    constexpr auto operator()(T1 &&t1) const noexcept(noexcept(detail::mem_ptr_target<C>(EINSUMS_FORWARD(T1, t1)).*pm))
        -> decltype(detail::mem_ptr_target<C>(EINSUMS_FORWARD(T1, t1)).*pm) {
        // This seems to trigger a bogus warning in GCC 11 with
        // optimizations enabled (possibly the same as this:
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98503) so we disable
        // the warning locally.
#if defined(EINSUMS_GCC_VERSION) && EINSUMS_GCC_VERSION >= 110000
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Warray-bounds"
#endif
        return detail::mem_ptr_target<C>(EINSUMS_FORWARD(T1, t1)).*pm;
#if defined(EINSUMS_GCC_VERSION) && EINSUMS_GCC_VERSION >= 110000
#    pragma GCC diagnostic pop
#endif
    }
};

template <typename T, typename C>
struct invoke_mem_fun {
    T C::*pm;

  public:
    constexpr invoke_mem_fun(T C::*pm) noexcept : pm(pm) {}

    template <typename T1, typename... Tn>
    constexpr auto operator()(T1 &&t1, Tn &&...tn) const
        noexcept(noexcept((detail::mem_ptr_target<C>(EINSUMS_FORWARD(T1, t1)).*pm)(EINSUMS_FORWARD(Tn, tn)...)))
            -> decltype((detail::mem_ptr_target<C>(EINSUMS_FORWARD(T1, t1)).*pm)(EINSUMS_FORWARD(Tn, tn)...)) {
        // This seems to trigger a bogus warning in GCC 11 with
        // optimizations enabled (possibly the same as this:
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98503) so we disable
        // the warning locally.
#if defined(EINSUMS_GCC_VERSION) && EINSUMS_GCC_VERSION >= 110000
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Warray-bounds"
#endif
        return (detail::mem_ptr_target<C>(EINSUMS_FORWARD(T1, t1)).*pm)(EINSUMS_FORWARD(Tn, tn)...);
#if defined(EINSUMS_GCC_VERSION) && EINSUMS_GCC_VERSION >= 110000
#    pragma GCC diagnostic pop
#endif
    }
};

template <typename F, typename FD = typename std::remove_cv<typename std::remove_reference<F>::type>::type>
struct dispatch_invoke {
    using type = F &&;
};

template <typename F, typename T, typename C>
struct dispatch_invoke<F, T C::*> {
    using type = std::conditional_t<std::is_function_v<T>, invoke_mem_fun<T, C>, invoke_mem_obj<T, C>>;
};

template <typename F>
using invoke_impl = typename dispatch_invoke<F>::type;

#define EINSUMS_INVOKE(F, ...) (::einsums::util::detail::invoke_impl<decltype((F))>(F)(__VA_ARGS__))

} // namespace einsums::util::detail