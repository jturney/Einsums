//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>
#else
// # include <einsums/concepts/concepts.hpp>
#    include <einsums/execution/algorithms/detail/partial_algorithm.hpp>
#    include <einsums/execution/algorithms/schedule_from.hpp>
#    include <einsums/execution_base/completion_scheduler.hpp>
#    include <einsums/execution_base/receiver.hpp>
#    include <einsums/execution_base/sender.hpp>
#    include <einsums/functional/detail/tag_priority_invoke.hpp>

#    include <utility>

namespace einsums::execution::experimental {
inline constexpr struct continues_on_t final : einsums::functional::detail::tag_priority<continues_on_t> {
  private:

        template <typename Sender, typename Scheduler>
            requires(
                is_sender_v<Sender> &&
                is_scheduler_v<Scheduler> &&
                einsums::execution::experimental::detail::
                    is_completion_scheduler_tag_invocable_v<
                        einsums::execution::experimental::set_value_t, Sender,
                        continues_on_t, Scheduler>)

    friend constexpr EINSUMS_FORCEINLINE auto tag_override_invoke(continues_on_t, Sender &&sender, Scheduler &&scheduler) {
        auto completion_scheduler =
            einsums::execution::experimental::get_completion_scheduler<einsums::execution::experimental::set_value_t>(
                einsums::execution::experimental::get_env(sender));
        return einsums::functional::detail::tag_invoke(continues_on_t{}, std::move(completion_scheduler), std::forward<Sender>(sender),
                                                       std::forward<Scheduler>(scheduler));
    }

    template <typename Sender, typename Scheduler>
        requires(is_sender_v<Sender> && is_scheduler_v<Scheduler>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(continues_on_t, Sender &&predecessor_sender, Scheduler &&scheduler) {
        return schedule_from(std::forward<Scheduler>(scheduler), std::forward<Sender>(predecessor_sender));
    }

    template <typename Scheduler>
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(continues_on_t, Scheduler &&scheduler) {
        return detail::partial_algorithm<continues_on_t, Scheduler>{std::forward<Scheduler>(scheduler)};
    }
} continues_on{};

using transfer_t EINSUMS_DEPRECATED("transfer_t has been renamed continues_on_t") = continues_on_t;
EINSUMS_DEPRECATED("transfer has been renamed continues_on")
inline constexpr continues_on_t transfer{};
} // namespace einsums::execution::experimental
#endif
