//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STDEXEC)
# include <einsums/execution_base/stdexec_forward.hpp>
#else
# include <einsums/execution/algorithms/continues_on.hpp>
# include <einsums/execution/algorithms/when_all.hpp>
# include <einsums/functional/detail/tag_fallback_invoke.hpp>

# include <utility>

namespace einsums::execution::experimental {
    EINSUMS_DEPRECATED("transfer_when_all will be removed in the future, use transfer and when_all "
                    "separately instead")
    inline constexpr struct transfer_when_all_t final
      : einsums::functional::detail::tag_fallback<transfer_when_all_t>
    {
    private:
        template <typename Scheduler, typename... Ts>
        friend constexpr EINSUMS_FORCEINLINE auto
        tag_fallback_invoke(transfer_when_all_t, Scheduler&& scheduler, Ts&&... ts)
        {
            return continues_on(
                when_all(std::forward<Ts>(ts)...), std::forward<Scheduler>(scheduler));
        }
    } transfer_when_all{};
}    // namespace einsums::execution::experimental
#endif
