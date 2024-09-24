//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>
#else
#    include <einsums/execution/algorithms/start_detached.hpp>
#    include <einsums/execution/algorithms/then.hpp>
#    include <einsums/execution_base/sender.hpp>
#    include <einsums/functional/detail/tag_fallback_invoke.hpp>

#    include <utility>

namespace einsums::execution::experimental {
inline constexpr struct execute_t final : einsums::functional::detail::tag_fallback<execute_t> {
  private:
    template <typename Scheduler, typename F>
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(execute_t, Scheduler &&scheduler, F &&f) {
        return start_detached(then(schedule(std::forward<Scheduler>(scheduler)), std::forward<F>(f)));
    }
} execute{};
} // namespace einsums::execution::experimental
#endif
