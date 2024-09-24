//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/type_support/type_identity.hpp>

#include <type_traits>
#include <utility>

namespace einsums::functional::detail {

template <auto &Tag>
using tag_t = std::decay_t<decltype(Tag)>;

namespace tag_invoke_t_ns {
void tag_invoke();

struct tag_invoke_t {
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename Tag, typename... Ts>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto
    EINSUMS_STATIC_CALL_OPERATOR(Tag tag, Ts &&...ts) noexcept(noexcept(tag_invoke(std::declval<Tag>(), std::forward<Ts>(ts)...)))
        -> decltype(tag_invoke(std::declval<Tag>(), std::forward<Ts>(ts)...)) {
        return tag_invoke(tag, std::forward<Ts>(ts)...);
    }

    friend constexpr auto operator==(tag_invoke_t, tag_invoke_t) -> bool { return true; }

    friend constexpr auto operator!=(tag_invoke_t, tag_invoke_t) -> bool { return false; }
};
} // namespace tag_invoke_t_ns

namespace tag_invoke_ns {
#if !defined(EINSUMS_COMPUTE_HOST_CODE)
inline constexpr tag_invoke_t_ns::tag_invoke_t tag_invoke = {};
#else
EINSUMS_DEVICE static tag_invoke_t_ns::tag_invoke_t tag_invoke = {};
#endif
} // namespace tag_invoke_ns

template <typename Tag, typename... Args>
using is_tag_invocable = std::is_invocable<decltype(tag_invoke_ns::tag_invoke), Tag, Args...>;

template <typename Tag, typename... Args>
inline constexpr bool is_tag_invocable_v = is_tag_invocable<Tag, Args...>::value;

namespace detail {
template <typename Sig, bool Dispatchable>
struct is_nothrow_tag_invocable_impl;

template <typename Sig>
struct is_nothrow_tag_invocable_impl<Sig, false> : std::false_type {};

template <typename Tag, typename... Args>
struct is_nothrow_tag_invocable_impl<decltype(tag_invoke_ns::tag_invoke)(Tag, Args...), true>
    : std::integral_constant<bool, noexcept(tag_invoke_ns::tag_invoke(std::declval<Tag>(), std::declval<Args>()...))> {};
} // namespace detail

// CUDA versions less than 11.2 have a template instantiation bug which
// leaves out certain template arguments and leads to us not being able to
// correctly check this condition. We default to the more relaxed
// noexcept(true) to not falsely exclude correct overloads. However, this
// may lead to noexcept(false) overloads falsely being candidates.
#if defined(__NVCC__) && defined(EINSUMS_CUDA_VERSION) && (EINSUMS_CUDA_VERSION < 1102)
template <typename Tag, typename... Args>
struct is_nothrow_tag_invocable : is_tag_invocable<Tag, Args...> {};
#else
template <typename Tag, typename... Args>
struct is_nothrow_tag_invocable
    : detail::is_nothrow_tag_invocable_impl<decltype(tag_invoke_ns::tag_invoke)(Tag, Args...), is_tag_invocable_v<Tag, Args...>> {};
#endif

template <typename Tag, typename... Args>
inline constexpr bool is_nothrow_tag_invocable_v = is_nothrow_tag_invocable<Tag, Args...>::value;

template <typename Tag, typename... Args>
using tag_invoke_result_t = decltype(tag_invoke_ns::tag_invoke(std::declval<Tag>(), std::declval<Args>()...));

template <typename Tag, typename... Args>
using tag_invoke_result = einsums::detail::type_identity<tag_invoke_result_t<Tag, Args...>>;

///////////////////////////////////////////////////////////////////////////////
namespace tag_base_ns {
// poison pill
void tag_invoke();

///////////////////////////////////////////////////////////////////////////
// helper base class implementing the tag_invoke logic for CPOs
template <typename Tag>
struct tag {
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto
    EINSUMS_STATIC_CALL_OPERATOR(Args &&...args) noexcept(is_nothrow_tag_invocable_v<Tag, Args...>) -> tag_invoke_result_t<Tag, Args...> {
        return tag_invoke(Tag{}, std::forward<Args>(args)...);
    }
};

template <typename Tag>
struct tag_noexcept {
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args>
        requires(is_nothrow_tag_invocable_v<Tag, Args...>)
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto EINSUMS_STATIC_CALL_OPERATOR(Args &&...args) noexcept
        -> tag_invoke_result_t<Tag, decltype(args)...> {
        return tag_invoke(Tag{}, std::forward<Args>(args)...);
    }
};
} // namespace tag_base_ns

inline namespace tag_invoke_base_ns {
template <typename Tag>
using tag = tag_base_ns::tag<Tag>;

template <typename Tag>
using tag_noexcept = tag_base_ns::tag_noexcept<Tag>;
} // namespace tag_invoke_base_ns

inline namespace tag_invoke_f_ns {
using tag_invoke_ns::tag_invoke;
} // namespace tag_invoke_f_ns
} // namespace einsums::functional::detail
