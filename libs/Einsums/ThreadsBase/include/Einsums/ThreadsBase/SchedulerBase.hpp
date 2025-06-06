//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/ThreadsBase/SchedulerMode.hpp>
#include <Einsums/ThreadsBase/ThreadPoolBase.hpp>

#include <atomic>

namespace einsums::threads::detail {

struct SchedulerBase {
    EINSUMS_NON_COPYABLE(SchedulerBase);

    SchedulerBase(std::size_t num_threads, char const *description = "", SchedulerMode mode = SchedulerMode::NothingSpecial);
    ~SchedulerBase();

    ThreadPoolBase *get_parent_pool() const {
        EINSUMS_ASSERT(_parent_pool != nullptr);
        return _parent_pool;
    }

    void set_parent_pool(ThreadPoolBase *p) {
        EINSUMS_ASSERT(_parent_pool == nullptr);
        _parent_pool = p;
    }

    std::size_t global_to_local_thread_index(std::size_t n) { return n - _parent_pool->get_thread_offset(); }

  protected:
    ThreadPoolBase *_parent_pool;
    char const     *_description;

    std::atomic<SchedulerMode> _mode;
};

} // namespace einsums::threads::detail