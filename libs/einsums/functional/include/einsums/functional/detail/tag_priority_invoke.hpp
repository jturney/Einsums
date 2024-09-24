//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#if defined(DOXYGEN)
namespace einsums::functional::detail {
inline namespace unspecified {
/// The `einsums::functional::detail::tag_override_invoke` name defines a constexpr object
/// that is invocable with one or more arguments. The first argument
/// is a 'tag' (typically a CPO). It is only invocable if an overload
/// of tag_override_invoke() that accepts the same arguments could be
/// found via ADL.
///
/// The evaluation of the expression
/// `einsums::functional::detail::tag_override_invoke(tag, args...)` is
/// equivalent to evaluating the unqualified call to
/// `tag_override_invoke(decay-copy(tag), std::forward<Args>args)...)`.
///
/// `einsums::functional::detail::tag_override_invoke` is implemented against P1895.
///
/// Example:
/// Defining a new customization point `foo`:
/// ```
/// namespace mylib {
///     inline constexpr
///         struct foo_fn final : einsums::functional::detail::tag_override<foo_fn>
///         {
///         } foo{};
/// }
/// ```
///
/// Defining an object `bar` which customizes `foo`:
/// ```
/// struct bar
/// {
///     int x = 42;
///
///     friend constexpr int tag_override_invoke(mylib::foo_fn, bar const& x)
///     {
///         return b.x;
///     }
/// };
/// ```
///
/// Using the customization point:
/// ```
/// static_assert(42 == mylib::foo(bar{}), "The answer is 42");
/// ```
inline constexpr unspecified tag_override_invoke = unspecified;
} // namespace unspecified

/// `einsums::functional::detail::is_tag_override_invocable<Tag, Args...>` is std::true_type if
/// an overload of `tag_override_invoke(tag, args...)` can be found via ADL.
template <typename Tag, typename... Args>
struct is_tag_override_invocable;

/// `einsums::functional::detail::is_tag_override_invocable_v<Tag, Args...>` evaluates to
/// `einsums::functional::detail::is_tag_override_invocable<Tag, Args...>::value`
template <typename Tag, typename... Args>
constexpr bool is_tag_override_invocable_v = is_tag_override_invocable<Tag, Args...>::value;

/// `einsums::functional::detail::is_nothrow_tag_override_invocable<Tag, Args...>` is
/// std::true_type if an overload of `tag_override_invoke(tag, args...)` can be
/// found via ADL and is noexcept.
template <typename Tag, typename... Args>
struct is_nothrow_tag_override_invocable;

/// `einsums::functional::detail::is_tag_override_invocable_v<Tag, Args...>` evaluates to
/// `einsums::functional::detail::is_tag_override_invocable<Tag, Args...>::value`
template <typename Tag, typename... Args>
constexpr bool is_nothrow_tag_override_invocable_v = is_nothrow_tag_override_invocable<Tag, Args...>::value;

/// `einsums::functional::detail::tag_override_invoke_result<Tag, Args...>` is the trait
/// returning the result type of the call einsums::functional::detail::tag_override_invoke. This
/// can be used in a SFINAE context.
template <typename Tag, typename... Args>
using tag_override_invoke_result = std::invoke_result<decltype(tag_override_invoke), Tag, Args...>;

/// `einsums::functional::detail::tag_override_invoke_result_t<Tag, Args...>` evaluates to
/// `einsums::functional::detail::tag_override_invoke_result_t<Tag, Args...>::type`
template <typename Tag, typename... Args>
using tag_override_invoke_result_t = typename tag_override_invoke_result<Tag, Args...>::type;

/// `einsums::functional::detail::tag_override<Tag>` defines a base class that implements
/// the necessary tag dispatching functionality for a given type `Tag`
template <typename Tag>
struct tag_override;

/// `einsums::functional::detail::tag_override_noexcept<Tag>` defines a base class that implements
/// the necessary tag dispatching functionality for a given type `Tag`
/// where the implementation is required to be noexcept
template <typename Tag>
struct tag_override_noexcept;
} // namespace einsums::functional::detail
#else

#    include <einsums/config.hpp>

#    include <einsums/functional/detail/tag_fallback_invoke.hpp>
#    include <einsums/functional/tag_invoke.hpp>
#    include <einsums/type_support/type_identity.hpp>

#    include <type_traits>
#    include <utility>

namespace einsums::functional::detail {
namespace tag_override_invoke_t_ns {

// poison pill
void tag_override_invoke();

struct tag_override_invoke_t {
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename Tag, typename... Ts>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto
    EINSUMS_STATIC_CALL_OPERATOR(Tag tag, Ts &&...ts) noexcept(noexcept(tag_override_invoke(std::declval<Tag>(), std::forward<Ts>(ts)...)))
        -> decltype(tag_override_invoke(std::declval<Tag>(), std::forward<Ts>(ts)...)) {
        return tag_override_invoke(tag, std::forward<Ts>(ts)...);
    }

    friend constexpr bool operator==(tag_override_invoke_t, tag_override_invoke_t) { return true; }

    friend constexpr bool operator!=(tag_override_invoke_t, tag_override_invoke_t) { return false; }
};
} // namespace tag_override_invoke_t_ns

namespace tag_override_invoke_ns {
#    if !defined(EINSUMS_COMPUTE_DEVICE_CODE)
inline constexpr tag_override_invoke_t_ns::tag_override_invoke_t tag_override_invoke = {};
#    else
EINSUMS_DEVICE static tag_override_invoke_t_ns::tag_override_invoke_t const tag_override_invoke = {};
#    endif
} // namespace tag_override_invoke_ns

///////////////////////////////////////////////////////////////////////////
template <typename Tag, typename... Args>
using is_tag_override_invocable = std::is_invocable<decltype(tag_override_invoke_ns::tag_override_invoke), Tag, Args...>;

template <typename Tag, typename... Args>
inline constexpr bool is_tag_override_invocable_v = is_tag_override_invocable<Tag, Args...>::value;

template <typename Sig, bool Dispatchable>
struct is_nothrow_tag_override_invocable_impl;

template <typename Sig>
struct is_nothrow_tag_override_invocable_impl<Sig, false> : std::false_type {};

template <typename Tag, typename... Args>
struct is_nothrow_tag_override_invocable_impl<decltype(tag_override_invoke_ns::tag_override_invoke)(Tag, Args...), true>
    : std::integral_constant<bool, noexcept(tag_override_invoke_ns::tag_override_invoke(std::declval<Tag>(), std::declval<Args>()...))> {};

// CUDA versions less than 11.2 have a template instantiation bug which
// leaves out certain template arguments and leads to us not being able to
// correctly check this condition. We default to the more relaxed
// noexcept(true) to not falsely exclude correct overloads. However, this
// may lead to noexcept(false) overloads falsely being candidates.
#    if defined(__NVCC__) && defined(EINSUMS_CUDA_VERSION) && (EINSUMS_CUDA_VERSION < 1102)
template <typename Tag, typename... Args>
struct is_nothrow_tag_override_invocable : std::true_type {};
#    else
template <typename Tag, typename... Args>
struct is_nothrow_tag_override_invocable
    : is_nothrow_tag_override_invocable_impl<decltype(tag_override_invoke_ns::tag_override_invoke)(Tag, Args...),
                                             is_tag_override_invocable_v<Tag, Args...>> {};
#    endif

template <typename Tag, typename... Args>
inline constexpr bool is_nothrow_tag_override_invocable_v = is_nothrow_tag_override_invocable<Tag, Args...>::value;

template <typename Tag, typename... Args>
using tag_override_invoke_result_t = decltype(tag_override_invoke_ns::tag_override_invoke(std::declval<Tag>(), std::declval<Args>()...));

template <typename Tag, typename... Args>
using tag_override_invoke_result = einsums::detail::type_identity<tag_override_invoke_result_t<Tag, Args...>>;

namespace tag_base_ns {
// poison pill
void tag_override_invoke();

///////////////////////////////////////////////////////////////////////////
/// Helper base class implementing the tag_invoke logic for CPOs that allow
/// overriding user-defined tag_invoke overloads with tag_override_invoke,
/// and that allow setting a fallback with tag_fallback_invoke.
///
/// This helper class is otherwise identical to tag_fallback, but allows
/// defining an implementation that will always take priority if it is
/// feasible. This is useful for example in cases where a member function
/// should always take priority over any free function tag_invoke overloads,
/// when available, like this:
///
/// template <typename T>
/// auto tag_override_invoke(T&& t) -> decltype(t.foo()){ return t.foo(); }
template <typename Tag>
struct tag_priority {
    // Is tag-override-dispatchable
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args, typename Enable = std::enable_if_t<is_tag_override_invocable_v<Tag, Args &&...>>>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto
    EINSUMS_STATIC_CALL_OPERATOR(Args &&...args) noexcept(is_nothrow_tag_override_invocable_v<Tag, Args...>)
        -> tag_override_invoke_result_t<Tag, Args &&...> {
        return tag_override_invoke(Tag{}, std::forward<Args>(args)...);
    }

    // Is not tag-override-dispatchable, but tag-dispatchable
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args,
              typename Enable = std::enable_if_t<!is_tag_override_invocable_v<Tag, Args &&...> && is_tag_invocable_v<Tag, Args &&...>>>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto
    EINSUMS_STATIC_CALL_OPERATOR(Args &&...args) noexcept(is_nothrow_tag_invocable_v<Tag, Args...>)
        -> tag_invoke_result_t<Tag, Args &&...> {
        return tag_invoke(Tag{}, std::forward<Args>(args)...);
    }

    // Is not tag-override-dispatchable, not tag-dispatchable, but
    // tag-fallback-dispatchable
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args,
              typename Enable = std::enable_if_t<!is_tag_override_invocable_v<Tag, Args &&...> && !is_tag_invocable_v<Tag, Args &&...> &&
                                                 is_tag_fallback_invocable_v<Tag, Args &&...>>>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto
    EINSUMS_STATIC_CALL_OPERATOR(Args &&...args) noexcept(is_nothrow_tag_fallback_invocable_v<Tag, Args...>)
        -> tag_fallback_invoke_result_t<Tag, Args &&...> {
        return tag_fallback_invoke(Tag{}, std::forward<Args>(args)...);
    }
};

///////////////////////////////////////////////////////////////////////////
// Helper base class implementing the tag_invoke logic for noexcept CPOs
// that allow overriding user-defined tag_invoke overloads with
// tag_override_invoke, and that allow setting a fallback with
// tag_fallback_invoke.
template <typename Tag>
struct tag_priority_noexcept {
    // Is nothrow tag-override-dispatchable
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args, typename Enable = std::enable_if_t<is_nothrow_tag_override_invocable_v<Tag, Args &&...>>>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto EINSUMS_STATIC_CALL_OPERATOR(Args &&...args) noexcept
        -> tag_override_invoke_result_t<Tag, Args &&...> {
        return tag_override_invoke(Tag{}, std::forward<Args>(args)...);
    }

    // Is not nothrow tag-override-dispatchable, but nothrow
    // tag-dispatchable
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args, typename Enable = std::enable_if_t<!is_nothrow_tag_override_invocable_v<Tag, Args &&...> &&
                                                                   is_nothrow_tag_invocable_v<Tag, Args &&...>>>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto EINSUMS_STATIC_CALL_OPERATOR(Args &&...args) noexcept
        -> tag_invoke_result_t<Tag, Args &&...> {
        return tag_invoke(Tag{}, std::forward<Args>(args)...);
    }

    // Is not nothrow tag-override-dispatchable, not nothrow
    // tag-dispatchable, but nothrow tag-fallback-dispatchable
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args, typename Enable = std::enable_if_t<!is_nothrow_tag_override_invocable_v<Tag, Args &&...> &&
                                                                   !is_nothrow_tag_invocable_v<Tag, Args &&...> &&
                                                                   is_nothrow_tag_fallback_invocable_v<Tag, Args &&...>>>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto EINSUMS_STATIC_CALL_OPERATOR(Args &&...args) noexcept
        -> tag_fallback_invoke_result_t<Tag, Args &&...> {
        return tag_fallback_invoke(Tag{}, std::forward<Args>(args)...);
    }
};
} // namespace tag_base_ns

inline namespace tag_invoke_base_ns {
template <typename Tag>
using tag_priority = tag_base_ns::tag_priority<Tag>;

template <typename Tag>
using tag_priority_noexcept = tag_base_ns::tag_priority_noexcept<Tag>;
} // namespace tag_invoke_base_ns

inline namespace tag_override_invoke_f_ns {
using tag_override_invoke_ns::tag_override_invoke;
} // namespace tag_override_invoke_f_ns
} // namespace einsums::functional::detail

#endif
