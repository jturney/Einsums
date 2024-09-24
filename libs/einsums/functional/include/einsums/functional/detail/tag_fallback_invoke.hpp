//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#if defined(DOXYGEN)
namespace einsums::functional::detail {
inline namespace unspecified {
/// The `einsums::functional::detail::tag_fallback_invoke` name defines a constexpr object
/// that is invocable with one or more arguments. The first argument
/// is a 'tag' (typically a CPO). It is only invocable if an overload
/// of tag_fallback_invoke() that accepts the same arguments could be
/// found via ADL.
///
/// The evaluation of the expression
/// `einsums::functional::detail::tag_fallback_invoke(tag, args...)` is
/// equivalent to evaluating the unqualified call to
/// `tag_fallback_invoke(decay-copy(tag), std::forward<Args>args)...)`.
///
/// `einsums::functional::detail::tag_fallback_invoke` is implemented against P1895.
///
/// Example:
/// Defining a new customization point `foo`:
/// ```
/// namespace mylib {
///     inline constexpr
///         struct foo_fn final : einsums::functional::detail::tag_fallback<foo_fn>
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
///     friend constexpr int tag_fallback_invoke(mylib::foo_fn, bar const& x)
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
inline constexpr unspecified tag_fallback_invoke = unspecified;
} // namespace unspecified

/// `einsums::functional::detail::is_tag_fallback_invocable<Tag, Args...>` is std::true_type if
/// an overload of `tag_fallback_invoke(tag, args...)` can be found via ADL.
template <typename Tag, typename... Args>
struct is_tag_fallback_invocable;

/// `einsums::functional::detail::is_tag_fallback_invocable_v<Tag, Args...>` evaluates to
/// `einsums::functional::detail::is_tag_fallback_invocable<Tag, Args...>::value`
template <typename Tag, typename... Args>
constexpr bool is_tag_fallback_invocable_v = is_tag_fallback_invocable<Tag, Args...>::value;

/// `einsums::functional::detail::is_nothrow_tag_fallback_invocable<Tag, Args...>` is
/// std::true_type if an overload of `tag_fallback_invoke(tag, args...)` can be
/// found via ADL and is noexcept.
template <typename Tag, typename... Args>
struct is_nothrow_tag_fallback_invocable;

/// `einsums::functional::detail::is_tag_fallback_invocable_v<Tag, Args...>` evaluates to
/// `einsums::functional::detail::is_tag_fallback_invocable<Tag, Args...>::value`
template <typename Tag, typename... Args>
constexpr bool is_nothrow_tag_fallback_invocable_v = is_nothrow_tag_fallback_invocable<Tag, Args...>::value;

/// `einsums::functional::detail::tag_fallback_invoke_result<Tag, Args...>` is the trait
/// returning the result type of the call einsums::functional::detail::tag_fallback_invoke. This
/// can be used in a SFINAE context.
template <typename Tag, typename... Args>
using tag_fallback_invoke_result = std::invoke_result<decltype(tag_fallback_invoke), Tag, Args...>;

/// `einsums::functional::detail::tag_fallback_invoke_result_t<Tag, Args...>` evaluates to
/// `einsums::functional::detail::tag_fallback_invoke_result_t<Tag, Args...>::type`
template <typename Tag, typename... Args>
using tag_fallback_invoke_result_t = typename tag_fallback_invoke_result<Tag, Args...>::type;

/// `einsums::functional::detail::tag_fallback<Tag>` defines a base class that implements
/// the necessary tag dispatching functionality for a given type `Tag`
template <typename Tag>
struct tag_fallback;

/// `einsums::functional::detail::tag_fallback_noexcept<Tag>` defines a base class that implements
/// the necessary tag dispatching functionality for a given type `Tag`
/// where the implementation is required to be noexcept
template <typename Tag>
struct tag_fallback_noexcept;
} // namespace einsums::functional::detail
#else

#    include <einsums/config.hpp>

#    include <einsums/functional/tag_invoke.hpp>
#    include <einsums/type_support/type_identity.hpp>

#    include <type_traits>
#    include <utility>

namespace einsums::functional::detail {
namespace tag_fallback_invoke_t_ns {
// poison pill
void tag_fallback_invoke();

struct tag_fallback_invoke_t {
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename Tag, typename... Ts>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto
    EINSUMS_STATIC_CALL_OPERATOR(Tag tag, Ts &&...ts) noexcept(noexcept(tag_fallback_invoke(std::declval<Tag>(), std::forward<Ts>(ts)...)))
        -> decltype(tag_fallback_invoke(std::declval<Tag>(), std::forward<Ts>(ts)...)) {
        return tag_fallback_invoke(tag, std::forward<Ts>(ts)...);
    }

    friend constexpr auto operator==(tag_fallback_invoke_t, tag_fallback_invoke_t) -> bool { return true; }

    friend constexpr auto operator!=(tag_fallback_invoke_t, tag_fallback_invoke_t) -> bool { return false; }
};
} // namespace tag_fallback_invoke_t_ns

namespace tag_fallback_invoke_ns {
#    if !defined(EINSUMS_COMPUTE_DEVICE_CODE)
inline constexpr tag_fallback_invoke_t_ns::tag_fallback_invoke_t tag_fallback_invoke = {};
#    else
EINSUMS_DEVICE static tag_fallback_invoke_t_ns::tag_fallback_invoke_t const tag_fallback_invoke = {};
#    endif
} // namespace tag_fallback_invoke_ns

///////////////////////////////////////////////////////////////////////////
template <typename Tag, typename... Args>
using is_tag_fallback_invocable = std::is_invocable<decltype(tag_fallback_invoke_ns::tag_fallback_invoke), Tag, Args...>;

template <typename Tag, typename... Args>
inline constexpr bool is_tag_fallback_invocable_v = is_tag_fallback_invocable<Tag, Args...>::value;

template <typename Sig, bool Dispatchable>
struct is_nothrow_tag_fallback_invocable_impl;

template <typename Sig>
struct is_nothrow_tag_fallback_invocable_impl<Sig, false> : std::false_type {};

template <typename Tag, typename... Args>
struct is_nothrow_tag_fallback_invocable_impl<decltype(tag_fallback_invoke_ns::tag_fallback_invoke)(Tag, Args...), true>
    : std::integral_constant<bool, noexcept(tag_fallback_invoke_ns::tag_fallback_invoke(std::declval<Tag>(), std::declval<Args>()...))> {};

// CUDA versions less than 11.2 have a template instantiation bug which
// leaves out certain template arguments and leads to us not being able to
// correctly check this condition. We default to the more relaxed
// noexcept(true) to not falsely exclude correct overloads. However, this
// may lead to noexcept(false) overloads falsely being candidates.
#    if defined(__NVCC__) && defined(EINSUMS_CUDA_VERSION) && (EINSUMS_CUDA_VERSION < 1102)
template <typename Tag, typename... Args>
struct is_nothrow_tag_fallback_invocable : std::true_type {};
#    else
template <typename Tag, typename... Args>
struct is_nothrow_tag_fallback_invocable
    : is_nothrow_tag_fallback_invocable_impl<decltype(tag_fallback_invoke_ns::tag_fallback_invoke)(Tag, Args...),
                                             is_tag_fallback_invocable_v<Tag, Args...>> {};
#    endif

template <typename Tag, typename... Args>
inline constexpr bool is_nothrow_tag_fallback_invocable_v = is_nothrow_tag_fallback_invocable<Tag, Args...>::value;

template <typename Tag, typename... Args>
using tag_fallback_invoke_result_t = decltype(tag_fallback_invoke_ns::tag_fallback_invoke(std::declval<Tag>(), std::declval<Args>()...));

template <typename Tag, typename... Args>
using tag_fallback_invoke_result = einsums::detail::type_identity<tag_fallback_invoke_result_t<Tag, Args...>>;

///////////////////////////////////////////////////////////////////////////////
namespace tag_base_ns {
template <typename Tag, typename... Args>
struct not_tag_fallback_noexcept_dispatchable;

// poison pill
void tag_fallback_invoke();

///////////////////////////////////////////////////////////////////////////
/// Helper base class implementing the tag_invoke logic for CPOs that fall
/// back to directly invoke its fallback.
///
/// This base class is in many cases preferable to the plain tag base class.
/// With the normal tag base class a default, unconstrained, default
/// tag_invoke overload will take precedence over user-defined tag_invoke
/// overloads that are not perfect matches. For example, with a default
/// overload:
///
/// template <typename T> auto tag_invoke(tag_t, T&& t) {...}
///
/// and a user-defined overload in another namespace:
///
/// auto tag_invoke(my_type t)
///
/// the user-defined overload will only be considered when it is an exact
/// match. This means const and reference qualifiers must match exactly, and
/// conversions to a base class are not considered.
///
/// With tag_fallback one can define the default implementation in terms of
/// a tag_fallback_invoke overload instead of tag_invoke:
///
/// template <typename T> auto tag_fallback_invoke(tag_t, T&& t) {...}
///
/// With the same user-defined tag_invoke overload, the user-defined
/// overload will now be used if it is a match even if it isn't an exact
/// match.
/// This is because tag_fallback will dispatch to tag_fallback_invoke only
/// if there are no matching tag_invoke overloads.
template <typename Tag>
struct tag_fallback {
    // is tag-dispatchable
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args>
        requires(is_tag_invocable_v<Tag, Args && ...>)
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto
    EINSUMS_STATIC_CALL_OPERATOR(Args &&...args) noexcept(is_nothrow_tag_invocable_v<Tag, Args...>)
        -> tag_invoke_result_t<Tag, Args &&...> {
        return tag_invoke(Tag{}, std::forward<Args>(args)...);
    }

    // is not tag-dispatchable
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args>
        requires(!is_tag_invocable_v<Tag, Args && ...>)
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto
    EINSUMS_STATIC_CALL_OPERATOR(Args &&...args) noexcept(is_nothrow_tag_fallback_invocable_v<Tag, Args...>)
        -> tag_fallback_invoke_result_t<Tag, Args &&...> {
        return tag_fallback_invoke(Tag{}, std::forward<Args>(args)...);
    }
};

///////////////////////////////////////////////////////////////////////////
// helper base class implementing the tag_invoke logic for CPOs that fall
// back to directly invoke its fallback. Either invocation has to be noexcept.
template <typename Tag>
struct tag_fallback_noexcept {
  private:
    // is nothrow tag-fallback dispatchable
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args>
    EINSUMS_HOST_DEVICE constexpr static auto tag_fallback_invoke_impl(std::false_type, Args &&.../*args*/) noexcept
        -> not_tag_fallback_noexcept_dispatchable<Tag, Args...> {
        return not_tag_fallback_noexcept_dispatchable<Tag, Args...>{};
    }

    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr static auto tag_fallback_invoke_impl(std::true_type, Args &&...args) noexcept
        -> tag_fallback_invoke_result_t<Tag, Args &&...> {
        return tag_fallback_invoke(Tag{}, std::forward<Args>(args)...);
    }

  public:
    // is nothrow tag-dispatchable
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args>
        requires(is_nothrow_tag_invocable_v<Tag, Args && ...>)
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto EINSUMS_STATIC_CALL_OPERATOR(Args &&...args) noexcept
        -> tag_invoke_result_t<Tag, Args &&...> {
        return tag_invoke(Tag{}, std::forward<Args>(args)...);
    }

    // is not nothrow tag-dispatchable
    EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
    template <typename... Args, typename IsFallbackDispatchable = is_nothrow_tag_fallback_invocable<Tag, Args &&...>>
        requires(!is_nothrow_tag_invocable_v<Tag, Args && ...>)
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE constexpr auto EINSUMS_STATIC_CALL_OPERATOR(Args &&...args) noexcept
        -> decltype(tag_fallback_invoke_impl(IsFallbackDispatchable{}, std::forward<Args>(args)...)) {
        return tag_fallback_invoke_impl(IsFallbackDispatchable{}, std::forward<Args>(args)...);
    }
};
} // namespace tag_base_ns

inline namespace tag_invoke_base_ns {
template <typename Tag>
using tag_fallback = tag_base_ns::tag_fallback<Tag>;

template <typename Tag>
using tag_fallback_noexcept = tag_base_ns::tag_fallback_noexcept<Tag>;
} // namespace tag_invoke_base_ns

inline namespace tag_fallback_invoke_f_ns {
using tag_fallback_invoke_ns::tag_fallback_invoke;
} // namespace tag_fallback_invoke_f_ns
} // namespace einsums::functional::detail

#endif
