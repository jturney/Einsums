//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Errors.hpp>
#include <Einsums/Hardware/AffinityData.hpp>
#include <Einsums/Hardware/CPUMask.hpp>
#include <Einsums/Hardware/Topology.hpp>
#include <Einsums/ThreadsBase/CallbackNotifier.hpp>
#include <Einsums/ThreadsBase/RuntimeState.hpp>
#include <Einsums/ThreadsBase/SchedulerMode.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <string>

namespace einsums::threads::detail {

struct PoolIdType {
    PoolIdType(std::size_t index, std::string const &name) : _index(index), _name(name) {}

    std::size_t        index() const { return _index; }
    std::string const &name() const { return _name; }

  private:
    std::size_t const _index;
    std::string const _name;
};

struct ThreadPoolInitParameters {
    std::string                   _name;
    std::size_t                   _index;
    SchedulerMode                 _mode;
    std::size_t                   _num_threads;
    std::size_t                   _thread_offset;
    CallbackNotifier             &_notifier;
    hardware::AffinityData const &_affinity_data;

    ThreadPoolInitParameters(std::string const &name, std::size_t index, SchedulerMode mode, std::size_t num_threads,
                             std::size_t thread_offset, CallbackNotifier &notifier, hardware::AffinityData const &affinity_data)
        : _name(name), _index(index), _mode(mode), _num_threads(num_threads), _thread_offset(thread_offset), _notifier(notifier),
          _affinity_data(affinity_data) {}
};

struct EINSUMS_EXPORT ThreadPoolBase {
    ThreadPoolBase(ThreadPoolInitParameters const &init);
    ~ThreadPoolBase() = default;

    PoolIdType get_pool_id() const { return _id; }

    virtual void init(std::size_t num_threads, std::size_t threads_offset);

    virtual bool run(std::unique_lock<std::mutex> &l, std::size_t num_threads) = 0;

    virtual void stop(std::unique_lock<std::mutex> &l, bool blocking = true) = 0;

    virtual void wait()    = 0;
    virtual bool is_busy() = 0;
    virtual bool is_idle() = 0;

    virtual void print_pool(std::ostream &) = 0;

    /**
     * Suspends the given processing unit. Blocks until the processing unit
     * has been suspended.
     *
     * @param core The processing unit on the pool to be suspended. Processing units are indexed starting at 0.
     */
    virtual void suspend_processing_unit_direct(std::size_t core) = 0;

    /**
     * Resumes the given processing unit. Blocks until the processing unit has been resumed.
     *
     * @param core The processing unit on the pool to be resumed. Processing units are indexed starting from 0.
     */
    virtual void resume_processing_unit_direct(std::size_t core) = 0;

    /**
     * Resumes the thread pool. Blocks until all OS threads on the thread pool have been resumed.
     */
    virtual void resume_all() = 0;

    /**
     * Suspends the thread pool. Blocks until all OS threads on the thread pool have been suspended.
     *
     * \note A thread pool cannot be suspended from an einsums thread running on the pool itself.
     *
     * \throws runtime_error if called from an einsums thread which is running on the pool itself
     */
    virtual void suspend_all() = 0;

    virtual std::size_t get_os_thread_count() const = 0;

    virtual std::thread &get_os_thread_handle(std::size_t num_thread) = 0;

    virtual std::size_t get_active_os_thread_count() const;

    virtual void create_thread(ThreadInitData &data, ThreadIdRefType &id) = 0;

    virtual ThreadIdRefType create_work(ThreadInitData &data) = 0;

    virtual ThreadState set_state(ThreadIdType const &id, ThreadScheduleState new_state, ThreadRestartState new_state_ex,
                                  execution::thread_priority priority) = 0;

  protected:
    PoolIdType  _id;
    std::size_t _thread_offset;

    hardware::AffinityData const &_affinity_data;

    CallbackNotifier &_notifier;
};

} // namespace einsums::threads::detail

template <>
struct fmt::formatter<einsums::threads::detail::ThreadPoolBase> : formatter<std::string> {
    template <typename FormatContext>
    auto format(einsums::threads::detail::ThreadPoolBase const &thread_pool, FormatContext &ctx) const {
        auto id = thread_pool.get_pool_id();
        return formatter<std::string>::format(fmt::format("{}({})", id.name(), id.index()), ctx);
    }
};