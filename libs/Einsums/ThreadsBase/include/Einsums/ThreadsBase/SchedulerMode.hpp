//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <cstdint>

namespace einsums::threads {

/// This enumeration describes the possible modes of a scheduler.
enum class SchedulerMode : std::uint32_t {
    /// As the name suggests, this option can be used to disable all other options.
    NothingSpecial = 0x000,
    /// The kernel priority of the os-thread driving the scheduler will be
    /// reduced below normal.
    reduce_thread_priority = 0x001,
    /// This option allows for the scheduler to dynamically increase and
    /// reduce the number of processing units it runs on. Setting this value
    /// not succeed for schedulers that do not support this functionality.
    enable_elasticity = 0x002,
    /// This option allows schedulers that support work thread/stealing to
    /// enable/disable it
    enable_stealing = 0x004,
    /// This option allows schedulersthat support it to disallow stealing
    /// between numa domains
    enable_stealing_numa = 0x008,
    /// This option tells schedulersthat support it to add tasks round
    /// robin to queues on each core
    assign_work_round_robin = 0x010,
    /// This option tells schedulers that support it to add tasks round to
    /// the same core/queue that the parent task is running on
    assign_work_thread_parent = 0x020,
    /// This option tells schedulers that support it to always (try to)
    /// steal high priority tasks from other queues before finishing their
    /// own lower priority tasks
    steal_high_priority_first = 0x040,
    /// This option tells schedulers that support it to steal tasks only
    /// when their local queues are empty
    steal_after_local = 0x080,
    /// This option allows for certain schedulers to explicitly disable
    /// exponential idle-back off
    enable_idle_backoff = 0x100,

    /// This option represents the default mode.
    default_mode = reduce_thread_priority | enable_stealing | enable_stealing_numa | assign_work_round_robin | steal_after_local,
    /// This enables all available options.
    all_flags = reduce_thread_priority | enable_elasticity | enable_stealing | enable_stealing_numa | assign_work_round_robin |
                assign_work_thread_parent | steal_high_priority_first | steal_after_local | enable_idle_backoff
};

EINSUMS_EXPORT SchedulerMode operator&(SchedulerMode sched1, SchedulerMode sched2);

EINSUMS_EXPORT SchedulerMode operator|(SchedulerMode sched1, SchedulerMode sched2);

EINSUMS_EXPORT SchedulerMode operator~(SchedulerMode sched);

} // namespace einsums::threads