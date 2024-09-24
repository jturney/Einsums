//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>
#include <einsums/execution_base/receiver.hpp>
#include <einsums/type_support/pack.hpp>

#include <type_traits>
#include <variant>

namespace einsums::execution::experimental::detail {

    template <typename T>
    struct result_type_signature_helper
    {
        using type = einsums::execution::experimental::set_value_t(T);
    };

    template <>
    struct result_type_signature_helper<void>
    {
        using type = einsums::execution::experimental::set_value_t();
    };

    template <typename T>
    using result_type_signature_helper_t = typename result_type_signature_helper<T>::type;

    template <typename Variants>
    struct single_result
    {
        static_assert(sizeof(Variants) == 0,
            "expected a single variant with a single type in sender_traits<>::value_types");
    };

    template <>
    struct single_result<einsums::util::detail::pack<einsums::util::detail::pack<>>>
    {
        using type = void;
    };

    template <typename T>
    struct single_result<einsums::util::detail::pack<einsums::util::detail::pack<T>>>
    {
        using type = T;
    };

    template <typename T, typename U, typename... Ts>
    struct single_result<einsums::util::detail::pack<einsums::util::detail::pack<T, U, Ts...>>>
    {
        static_assert(sizeof(T) == 0,
            "expected a single variant with a single type in sender_traits<>::value_types (single "
            "variant with two or more types given)");
    };

    template <typename T, typename U, typename... Ts>
    struct single_result<einsums::util::detail::pack<T, U, Ts...>>
    {
        static_assert(sizeof(T) == 0,
            "expected a single variant with a single type in sender_traits<>::value_types (two or "
            "more variants)");
    };

    template <typename Variants>
    using single_result_t = typename single_result<Variants>::type;

    template <typename Variants>
    struct single_result_non_void
    {
        using type = typename single_result<Variants>::type;
        static_assert(!std::is_void<type>::value, "expected a non-void type in single_result");
    };

    template <typename Variants>
    using single_result_non_void_t = typename single_result_non_void<Variants>::type;

    template <typename Variants>
    struct single_variant
    {
        static_assert(
            sizeof(Variants) == 0, "expected a single variant sender_traits<>::value_types");
    };

    template <typename T>
    struct single_variant<einsums::util::detail::pack<T>>
    {
        using type = T;
    };

    template <typename Variants>
    using single_variant_t = typename single_variant<Variants>::type;

}    // namespace einsums::execution::experimental::detail
