//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <deque>
#    include <mutex>
#    include <string>
#    include <utility>
#    include <vector>

EINSUMS_NAMESPACE_BEGIN(profile)

/// One log message or program output line, queued for the viewer.
struct LogEntry {
    int         level;     // 0 trace, 1 debug, 2 info, 3 warning, 4 error, 5 critical (spdlog's numbering)
    std::string timestamp; // ISO 8601 with milliseconds
    std::string file;      // source file (basename)
    int         line;
    std::string function;
    std::string message; // formatted message text
};

class LogMessageQueue {
  public:
    static constexpr size_t kMaxPending = 1000;

    void push(LogEntry entry) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_queue.size() >= kMaxPending) {
            _queue.pop_front();
        }
        _queue.push_back(std::move(entry));
    }

    std::vector<LogEntry> drain() {
        std::lock_guard<std::mutex> lock(_mutex);
        std::vector<LogEntry>       result(std::make_move_iterator(_queue.begin()), std::make_move_iterator(_queue.end()));
        _queue.clear();
        return result;
    }

  private:
    std::mutex           _mutex;
    std::deque<LogEntry> _queue;
};

EINSUMS_NAMESPACE_END(profile)

#endif
