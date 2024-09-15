//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/type_support/void_guard.hpp>

#include <type_traits>
#include <utility>

namespace einsums::util::detail {
// when `pm` is a pointer to member of a class `C` and
// `is_base_of_v<C, remove_reference_t<T>>` is `true`;
template <typename C, typename T,
          typename = typename std::enable_if<std::is_base_of<C, typename std::remove_reference<T>::type>::value>::type>
static constexpr T &&mem_ptr_target(T &&v) noexcept {
    return std::forward<T>(v);
}

// when `pm` is a pointer to member of a class `C` and
// `remove_cvref_t<T>` is a specialization of `reference_wrapper`;
template <typename C, typename T>
static constexpr T &mem_ptr_target(std::reference_wrapper<T> v) noexcept {
    return v.get();
}

// when `pm` is a pointer to member of a class `C` and `T` does not
// satisfy the previous two items;
template <typename C, typename T>
static constexpr auto mem_ptr_target(T &&v) noexcept(noexcept(*std::forward<T>(v))) -> decltype(*std::forward<T>(v)) {
    return *std::forward<T>(v);
}

///////////////////////////////////////////////////////////////////////////
template <typename T, typename C>
struct invoke_mem_obj {
    T C::*pm;

  public:
    constexpr invoke_mem_obj(T C::*pm) noexcept : pm(pm) {}

    template <typename T1>
    constexpr auto operator()(T1 &&t1) const noexcept(noexcept(detail::mem_ptr_target<C>(std::forward<T1>(t1)).*
                                                               pm)) -> decltype(detail::mem_ptr_target<C>(std::forward<T1>(t1)).*pm) {
        // This seems to trigger a bogus warning in GCC 11 with
        // optimizations enabled (possibly the same as this:
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98503) so we disable
        // the warning locally.
#if defined(EINSUMS_GCC_VERSION) && EINSUMS_GCC_VERSION >= 110000
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Warray-bounds"
#endif
        return detail::mem_ptr_target<C>(std::forward<T1>(t1)).*pm;
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
        noexcept(noexcept((detail::mem_ptr_target<C>(std::forward<T1>(t1)).*pm)(std::forward<Tn>(tn)...)))
            -> decltype((detail::mem_ptr_target<C>(std::forward<T1>(t1)).*pm)(std::forward<Tn>(tn)...)) {
        // This seems to trigger a bogus warning in GCC 11 with
        // optimizations enabled (possibly the same as this:
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98503) so we disable
        // the warning locally.
#if defined(EINSUMS_GCC_VERSION) && EINSUMS_GCC_VERSION >= 110000
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Warray-bounds"
#endif
        return (detail::mem_ptr_target<C>(std::forward<T1>(t1)).*pm)(std::forward<Tn>(tn)...);
#if defined(EINSUMS_GCC_VERSION) && EINSUMS_GCC_VERSION >= 110000
#    pragma GCC diagnostic pop
#endif
    }
};

///////////////////////////////////////////////////////////////////////////
template <typename F, typename FD = typename std::remove_cv<typename std::remove_reference<F>::type>::type>
struct dispatch_invoke {
    using type = F &&;
};

template <typename F, typename T, typename C>
struct dispatch_invoke<F, T C::*> {
    using type = typename std::conditional<std::is_function<T>::value, invoke_mem_fun<T, C>, invoke_mem_obj<T, C>>::type;
};

template <typename F>
using invoke_impl = typename dispatch_invoke<F>::type;

#define EINSUMS_INVOKE(F, ...) (::einsums::util::detail::invoke_impl<decltype((F))>(F)(__VA_ARGS__))

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
constexpr EINSUMS_HOST_DEVICE std::invoke_result_t<F, Ts...> invoke(F &&f, Ts &&...vs) {
    return EINSUMS_INVOKE(std::forward<F>(f), std::forward<Ts>(vs)...);
}

///////////////////////////////////////////////////////////////////////////
/// \copydoc invoke
///
/// \tparam R The result type of the function when it's called
///           with the content of the given argument types vs.
template <typename R, typename F, typename... Ts>
constexpr EINSUMS_HOST_DEVICE R invoke_r(F &&f, Ts &&...vs) {
    return EINSUMS_INVOKE_R(R, std::forward<F>(f), std::forward<Ts>(vs)...);
}

} // namespace einsums::util::detail