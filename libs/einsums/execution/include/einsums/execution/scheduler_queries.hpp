//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>
#if defined(EINSUMS_HAVE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>
#else
// #    include <einsums/concepts/concepts.hpp>
#    include <einsums/execution_base/sender.hpp>
#    include <einsums/functional/tag_invoke.hpp>

namespace einsums::execution::experimental {
enum class forward_progress_guarantee { concurrent, parallel, weakly_parallel };

namespace scheduler_queries_detail {
struct forwarding_scheduler_query_t {
    template <typename Query>
        requires(einsums::functional::detail::is_nothrow_tag_invocable_v<forwarding_scheduler_query_t, Query const &>)
    constexpr bool EINSUMS_STATIC_CALL_OPERATOR(Query const &query) noexcept {
        return einsums::functional::detail::tag_invoke(forwarding_scheduler_query_t{}, query);
    }

    template <typename Query>
        requires(!einsums::functional::detail::is_nothrow_tag_invocable_v<forwarding_scheduler_query_t, Query const &>)
    constexpr bool EINSUMS_STATIC_CALL_OPERATOR(Query const &) noexcept {
        return false;
    }
};

struct get_forward_progress_guarantee_t {
    template <typename Scheduler>
        requires(is_scheduler_v<Scheduler> &&
                 einsums::functional::detail::is_nothrow_tag_invocable_v<get_forward_progress_guarantee_t, Scheduler const &>)
    constexpr forward_progress_guarantee EINSUMS_STATIC_CALL_OPERATOR(Scheduler const &scheduler) noexcept {
        return einsums::functional::detail::tag_invoke(get_forward_progress_guarantee_t{}, scheduler);
    }

    template <typename Scheduler>
        requires(is_scheduler_v<Scheduler> &&
                 !einsums::functional::detail::is_nothrow_tag_invocable_v<get_forward_progress_guarantee_t, Scheduler const &>)
    constexpr forward_progress_guarantee EINSUMS_STATIC_CALL_OPERATOR(Scheduler const &) noexcept {
        return forward_progress_guarantee::weakly_parallel;
    }
};
} // namespace scheduler_queries_detail

using scheduler_queries_detail::forwarding_scheduler_query_t;
using scheduler_queries_detail::get_forward_progress_guarantee_t;

inline constexpr forwarding_scheduler_query_t     forwarding_scheduler_query{};
inline constexpr get_forward_progress_guarantee_t get_forward_progress_guarantee{};
} // namespace einsums::execution::experimental
#endif
