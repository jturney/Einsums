//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace einsums::resource {

struct socket;
struct core;
struct pu;

struct partitioner;

namespace detail {
struct EINSUMS_EXPORT partitioner;
void EINSUMS_EXPORT   delete_partitioner();
} // namespace detail

EINSUMS_EXPORT detail::partitioner &get_partitioner();

EINSUMS_EXPORT bool is_partitioner_valid();

enum partitioner_mode { mode_default = 0, mode_allow_oversubscription = 1, mode_allow_dynamic_pools = 2 };

enum scheduling_policy {
    user_defined        = -2,
    unspecified         = -1,
    local               = 0,
    local_priority_fifo = 1,
    local_priority_lifo = 2,
    static_             = 3,
    static_priority     = 4,
    abp_priority_fifo   = 5,
    abp_priority_lifo   = 6,
    shared_priority     = 7,
};

namespace detail {
EINSUMS_EXPORT char const *get_scheduling_policy_name(scheduling_policy p) noexcept;
}

} // namespace einsums::resource

template <>
struct fmt::formatter<einsums::resource::scheduling_policy> : fmt::formatter<char const *> {
    template <typename FormatContext>
    auto format(einsums::resource::scheduling_policy p, FormatContext &ctx) const {
        return fmt::formatter<char const *>::format(einsums::resource::detail::get_scheduling_policy_name((p), ctx));
    }
};