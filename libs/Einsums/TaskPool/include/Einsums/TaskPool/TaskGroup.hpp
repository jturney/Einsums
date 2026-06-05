//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>

namespace einsums::task_pool {

/// @brief Handle for a group of submitted tasks with a collective barrier.
///
/// Created by TaskPool::submit_group(). Use wait_all() to block until
/// every task in the group has completed.
class TaskGroup {
  public:
    TaskGroup() = default;

    /// @brief Block until all tasks in the group complete.
    void wait_all() {
        std::unique_lock lock(*_mutex);
        // Use a timed wait to guard against lost notifications.
        // Workers have a 1ms park timeout, so 10ms per poll is more than enough.
        while (_remaining->load() != 0) {
            _cv->wait_for(lock, std::chrono::milliseconds(10));
        }
    }

    /// @brief Check if all tasks have completed (non-blocking).
    [[nodiscard]] bool ready() const { return _remaining->load() == 0; }

    /// @brief Number of tasks in the group.
    [[nodiscard]] size_t size() const { return _total; }

    /// @brief Number of tasks remaining.
    [[nodiscard]] size_t remaining() const { return _remaining->load(); }

    /// @brief Group name (for profiler).
    [[nodiscard]] std::string const &name() const { return _name; }

  private:
    friend class TaskPool;

    TaskGroup(std::string name, size_t total)
        : _name(std::move(name)), _total(total), _remaining(std::make_shared<std::atomic<size_t>>(total)),
          _mutex(std::make_shared<std::mutex>()), _cv(std::make_shared<std::condition_variable>()) {}

    /// Called by each task on completion.
    void notify_one_complete() {
        if (_remaining->fetch_sub(1) == 1) {
            _cv->notify_all();
        }
    }

    std::string                              _name;
    size_t                                   _total{0};
    std::shared_ptr<std::atomic<size_t>>     _remaining;
    std::shared_ptr<std::mutex>              _mutex;
    std::shared_ptr<std::condition_variable> _cv;
};

} // namespace einsums::task_pool
