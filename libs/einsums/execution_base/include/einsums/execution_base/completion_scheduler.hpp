//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STDEXEC)
# include <einsums/execution_base/sender.hpp>
# include <einsums/execution_base/stdexec_forward.hpp>
# include <einsums/functional/detail/invoke_result_plain_function.hpp>
# include <einsums/functional/tag_invoke.hpp>

namespace einsums::execution::experimental::detail {
    template <bool TagInvocable, typename CPO, typename Sender>
    struct has_completion_scheduler_impl : std::false_type
    {
    };

    template <typename CPO, typename Sender>
    struct has_completion_scheduler_impl<true, CPO, Sender>
      : einsums::execution::experimental::is_scheduler<einsums::detail::invoke_result_plain_function_t<
            get_completion_scheduler_t<CPO>,
            einsums::detail::invoke_result_plain_function_t<get_env_t, std::decay_t<Sender> const&>>>
    {
    };

    template <typename CPO, typename Sender>
    struct has_completion_scheduler
      : has_completion_scheduler_impl<
            einsums::functional::detail::is_tag_invocable_v<get_completion_scheduler_t<CPO>,
                einsums::detail::invoke_result_plain_function_t<get_env_t,
                    std::decay_t<Sender> const&>>,
            CPO, Sender>
    {
    };

    template <typename CPO, typename Sender>
    inline constexpr bool has_completion_scheduler_v = has_completion_scheduler<CPO, Sender>::value;
}    // namespace einsums::execution::experimental::detail
#else
# include <einsums/execution_base/sender.hpp>
# include <einsums/functional/tag_invoke.hpp>

# include <type_traits>

namespace einsums::execution::experimental {
    template <typename CPO>
    struct get_completion_scheduler_t final
      : einsums::functional::detail::tag<get_completion_scheduler_t<CPO>>
    {
    };

    template <typename CPO>
    inline constexpr get_completion_scheduler_t<CPO> get_completion_scheduler{};

    namespace detail {
        template <bool TagInvocable, typename CPO, typename Sender>
        struct has_completion_scheduler_impl : std::false_type
        {
        };

        template <typename CPO, typename Sender>
        struct has_completion_scheduler_impl<true, CPO, Sender>
          : einsums::execution::experimental::is_scheduler<
                einsums::detail::invoke_result_plain_function_t<get_completion_scheduler_t<CPO>,
                    einsums::detail::invoke_result_plain_function_t<get_env_t,
                        std::decay_t<Sender> const&>>>
        {
        };

        template <typename CPO, typename Sender>
        struct has_completion_scheduler
          : has_completion_scheduler_impl<std::is_invocable_v<get_completion_scheduler_t<CPO>,
                                              einsums::detail::invoke_result_plain_function_t<
                                                  get_env_t, std::decay_t<Sender> const&>>,
                CPO, Sender>
        {
        };

        template <typename CPO, typename Sender>
        inline constexpr bool has_completion_scheduler_v =
            has_completion_scheduler<CPO, Sender>::value;

        template <bool HasCompletionScheduler, typename ReceiverCPO, typename Sender,
            typename AlgorithmCPO, typename... Ts>
        struct is_completion_scheduler_tag_invocable_impl : std::false_type
        {
        };

        template <typename ReceiverCPO, typename Sender, typename AlgorithmCPO, typename... Ts>
        struct is_completion_scheduler_tag_invocable_impl<true, ReceiverCPO, Sender, AlgorithmCPO,
            Ts...>
          : std::integral_constant<bool,
                std::is_invocable_v<AlgorithmCPO,
                    einsums::detail::invoke_result_plain_function_t<
                        einsums::execution::experimental::get_completion_scheduler_t<ReceiverCPO>,
                        einsums::detail::invoke_result_plain_function_t<
                            einsums::execution::experimental::get_env_t, std::decay_t<Sender> const&>>,
                    Sender, Ts...>>
        {
        };

        template <typename ReceiverCPO, typename Sender, typename AlgorithmCPO, typename... Ts>
        struct is_completion_scheduler_tag_invocable
          : is_completion_scheduler_tag_invocable_impl<
                einsums::execution::experimental::detail::has_completion_scheduler_v<ReceiverCPO,
                    Sender>,
                ReceiverCPO, Sender, AlgorithmCPO, Ts...>
        {
        };

        template <typename ReceiverCPO, typename Sender, typename AlgorithmCPO, typename... Ts>
        inline constexpr bool is_completion_scheduler_tag_invocable_v =
            is_completion_scheduler_tag_invocable<ReceiverCPO, Sender, AlgorithmCPO, Ts...>::value;

    }    // namespace detail
}    // namespace einsums::execution::experimental
#endif
