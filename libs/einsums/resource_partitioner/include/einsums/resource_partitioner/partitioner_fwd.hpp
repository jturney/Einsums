//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/function.hpp>
#include <einsums/threading_base/thread_pool_base.hpp>
#include <einsums/threading_base/thread_queue_init_parameters.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <memory>
#include <string>

namespace einsums::resource {
class socket;
class core;
class pu;

class partitioner;

namespace detail {
class EINSUMS_EXPORT partitioner;
void EINSUMS_EXPORT  delete_partitioner();
} // namespace detail

/// May be used anywhere in code and returns a reference to the single,
/// global resource partitioner.
EINSUMS_EXPORT detail::partitioner &get_partitioner();

/// Returns true if the resource partitioner has been initialized.
/// Returns false otherwise.
EINSUMS_EXPORT bool is_partitioner_valid();

/// This enumeration describes the modes available when creating a
/// resource partitioner.
enum partitioner_mode {
    /// Default mode.
    mode_default = 0,
    /// Allow processing units to be oversubscribed, i.e. multiple
    /// worker threads to share a single processing unit.
    mode_allow_oversubscription = 1,
    /// Allow worker threads to be added and removed from thread pools.
    mode_allow_dynamic_pools = 2
};

using scheduler_function = util::detail::function<std::unique_ptr<einsums::threads::detail::thread_pool_base>(
    einsums::threads::detail::thread_pool_init_parameters, einsums::threads::detail::thread_queue_init_parameters)>;

// Choose same names as in command-line options except with _ instead of
// -.

/// This enumeration lists the available scheduling policies (or
/// schedulers) when creating thread pools.
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
        return fmt::formatter<char const *>::format(einsums::resource::detail::get_scheduling_policy_name(p), ctx);
    }
};
