//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <chrono>
#    include <deque>
#    include <filesystem>
#    include <mutex>
#    include <spdlog/sinks/base_sink.h>
#    include <string>
#    include <vector>

namespace einsums::profile {

struct LogEntry {
    int         level;     // spdlog::level::level_enum as int
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

// ─── Benchmark result queue ──────────────────────────────────────────────────

/// A benchmark result entry sent via the profiler server as type="benchmark_result".
struct BenchmarkResultEntry {
    std::string label;    ///< e.g., "gemm N=256"
    std::string metric;   ///< e.g., "t_einsum", "t_generic"
    double      value_us; ///< Primary timing (average)
    double      min_us;
    double      max_us;
    double      stddev_us;
    double      warmup_us;
    int         reps;
    std::string annotations_json; ///< Pre-serialized JSON object of annotations
};

class BenchmarkResultQueue {
  public:
    static constexpr size_t kMaxPending = 10000;

    void push(BenchmarkResultEntry entry) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_queue.size() >= kMaxPending)
            _queue.pop_front();
        _queue.push_back(std::move(entry));
    }

    std::vector<BenchmarkResultEntry> drain() {
        std::lock_guard<std::mutex>       lock(_mutex);
        std::vector<BenchmarkResultEntry> result(std::make_move_iterator(_queue.begin()), std::make_move_iterator(_queue.end()));
        _queue.clear();
        return result;
    }

  private:
    std::mutex                       _mutex;
    std::deque<BenchmarkResultEntry> _queue;
};

/// spdlog sink that forwards log messages to a LogMessageQueue.
/// IMPORTANT: This sink must never call EINSUMS_LOG_* to avoid recursion.
class ProfilerSink : public spdlog::sinks::base_sink<std::mutex> {
  public:
    explicit ProfilerSink(LogMessageQueue *queue) : _queue(queue) {}

  protected:
    void sink_it_(spdlog::details::log_msg const &msg) override {
        // Extract the raw message payload (no ANSI, no prefix)
        std::string message_text(msg.payload.data(), msg.payload.size());

        // Format timestamp as ISO 8601 with milliseconds
        auto    time_point = msg.time;
        auto    time_t_val = std::chrono::system_clock::to_time_t(time_point);
        auto    ms         = std::chrono::duration_cast<std::chrono::milliseconds>(time_point.time_since_epoch()) % 1000;
        std::tm tm{};
#    ifdef _WIN32
        localtime_s(&tm, &time_t_val);
#    else
        localtime_r(&time_t_val, &tm);
#    endif
        char ts_buf[32];
        std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%S", &tm);
        char ms_buf[8];
        std::snprintf(ms_buf, sizeof(ms_buf), ".%03d", static_cast<int>(ms.count()));
        std::string timestamp = std::string(ts_buf) + ms_buf;

        // Extract source location
        std::string file;
        int         line = 0;
        std::string function;
        if (msg.source.filename) {
            file = std::filesystem::path(msg.source.filename).filename().string();
        }
        if (msg.source.line > 0) {
            line = msg.source.line;
        }
        if (msg.source.funcname) {
            function = msg.source.funcname;
        }

        LogEntry entry;
        entry.level     = static_cast<int>(msg.level);
        entry.timestamp = std::move(timestamp);
        entry.file      = std::move(file);
        entry.line      = line;
        entry.function  = std::move(function);
        entry.message   = std::move(message_text);

        _queue->push(std::move(entry));
    }

    void flush_() override {
        // no-op
    }

  private:
    LogMessageQueue *_queue;
};

using ProfilerSinkMt = ProfilerSink;

} // namespace einsums::profile

#endif
