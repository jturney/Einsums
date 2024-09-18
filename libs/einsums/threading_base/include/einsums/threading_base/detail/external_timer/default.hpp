//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/coroutines/thread_id_type.hpp>
#include <einsums/threading_base/thread_description.hpp>
#include <einsums/threading_base/threading_base_fwd.hpp>

#include <cstdint>
#include <memory>

namespace einsums::detail::external_timer {
inline std::shared_ptr<task_wrapper> new_task(einsums::detail::thread_description const &, std::uint32_t, threads::detail::thread_id_type) {
    return nullptr;
}

inline std::shared_ptr<task_wrapper> update_task(std::shared_ptr<task_wrapper>, einsums::detail::thread_description const &) {
    return nullptr;
}

struct [[nodiscard]] scoped_timer {
    explicit scoped_timer(std::shared_ptr<task_wrapper>){};
    scoped_timer(scoped_timer &&)                 = delete;
    scoped_timer(scoped_timer const &)            = delete;
    scoped_timer &operator=(scoped_timer &&)      = delete;
    scoped_timer &operator=(scoped_timer const &) = delete;
    ~scoped_timer()                               = default;

    void stop() {}
    void yield() {}
};
} // namespace einsums::detail::external_timer
