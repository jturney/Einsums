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
# include <einsums/execution/algorithms/just.hpp>
# include <einsums/functional/detail/tag_fallback_invoke.hpp>

# include <utility>

namespace einsums::execution::experimental {
    EINSUMS_DEPRECATED(
        "transfer_just will be removed in the future, use transfer and just separately instead")
    inline constexpr struct transfer_just_t final
      : einsums::functional::detail::tag_fallback<transfer_just_t>
    {
    private:
        template <typename Scheduler, typename... Ts>
        friend constexpr EINSUMS_FORCEINLINE auto
        tag_fallback_invoke(transfer_just_t, Scheduler&& scheduler, Ts&&... ts)
        {
            return continues_on(just(std::forward<Ts>(ts)...), std::forward<Scheduler>(scheduler));
        }
    } transfer_just{};
}    // namespace einsums::execution::experimental
#endif
