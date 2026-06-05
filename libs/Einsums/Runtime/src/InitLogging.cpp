//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Logging.hpp>
#include <Einsums/Runtime/Detail/InitLogging.hpp>
#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>

#if defined(EINSUMS_HAVE_PROFILER)
#    include <Einsums/Profile/LogSink.hpp>
#    include <Einsums/Profile/Profile.hpp>
#endif

#include <fmt/format.h>

#include <chrono>
#include <ctime>
#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>
#include <string>

namespace einsums::detail {

namespace {
/*
 * The desired output is "tid/description". Modern operating systems
 * allow you to set a description on a thread that can be accessed.
 * This could be useful in the future when we introduce thread pools
 * The logger can output the thread description to aid in debugging.
 */
void spdlog_format_thread_id(int pid, spdlog::details::log_msg const &, std::tm const &, spdlog::memory_buf_t &dest) {
    dest.append(fmt::format("{}/----", pid));
}
} // namespace

struct ThreadIdFormatterFlag : spdlog::custom_flag_formatter {
    void format(spdlog::details::log_msg const &msg, std::tm const &tm_time, spdlog::memory_buf_t &dest) override {
        spdlog_format_thread_id(getpid(), msg, tm_time, dest);
    }

    [[nodiscard]] std::unique_ptr<custom_flag_formatter> clone() const override {
        return spdlog::details::make_unique<ThreadIdFormatterFlag>();
    }
};

struct ParentThreadIdFormatterFlag : spdlog::custom_flag_formatter {
    void format(spdlog::details::log_msg const &msg, std::tm const &tm_time, spdlog::memory_buf_t &dest) override {
#if defined(EINSUMS_WINDOWS)
        /// @todo There is a way to get the parent pid on Windows. Just don't want to do it now.
        spdlog_format_thread_id(0, msg, tm_time, dest);
#else
        spdlog_format_thread_id(getppid(), msg, tm_time, dest);
#endif
    }

    [[nodiscard]] std::unique_ptr<custom_flag_formatter> clone() const override {
        return spdlog::details::make_unique<ParentThreadIdFormatterFlag>();
    }
};

/*
 * The desired output is "hostname" or eventually when MPI is added
 * "hostname/rank".
 */
struct HostnameFormatterFlag : spdlog::custom_flag_formatter {
    void format(spdlog::details::log_msg const &msg, std::tm const &tm_time, spdlog::memory_buf_t &dest) override {
        dest.append(std::string("localhost"));
    }

    [[nodiscard]] std::unique_ptr<custom_flag_formatter> clone() const override {
        return spdlog::details::make_unique<HostnameFormatterFlag>();
    }
};

void init_logging(RuntimeConfiguration &config) {
    auto &global_config = GlobalConfigMap::get_singleton();
    // Set log destination
    auto &sinks = get_einsums_logger().sinks();
    sinks.clear();
    sinks.push_back(get_spdlog_sink(global_config.get_string("log-destination")));
#if defined(EINSUMS_HAVE_TRACY)
    sinks.push_back(get_spdlog_sink("tracy"));
#endif

    // Set log pattern
    auto formatter = std::make_unique<spdlog::pattern_formatter>();
    formatter->add_flag<ThreadIdFormatterFlag>('k');
    formatter->add_flag<ParentThreadIdFormatterFlag>('q');
    formatter->add_flag<HostnameFormatterFlag>('j');
    formatter->set_pattern(global_config.get_string("log-format"));
    get_einsums_logger().set_formatter(std::move(formatter));

    // Set log level
    get_einsums_logger().set_level(static_cast<spdlog::level::level_enum>(global_config.get_int("log-level")));
    // global_config.attach(handle_loglevel_changes);

#if defined(EINSUMS_HAVE_PROFILER)
    {
        auto *srv = profile::Profiler::instance().server();
        if (srv) {
            auto profiler_sink_ptr = std::make_shared<profile::ProfilerSinkMt>(&srv->log_queue());
            profiler_sink_ptr->set_level(spdlog::level::trace);
            sinks.push_back(profiler_sink_ptr);

            // Wire println output to the profiler TCP server
            auto *output_q = &srv->output_queue();
            einsums::print::set_output_sink([output_q](std::string const &msg) {
                profile::LogEntry entry;
                entry.level = 2; // INFO-equivalent

                // Format timestamp as ISO 8601 with milliseconds
                auto    now        = std::chrono::system_clock::now();
                auto    time_t_val = std::chrono::system_clock::to_time_t(now);
                auto    ms         = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
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
                entry.timestamp = std::string(ts_buf) + ms_buf;

                entry.message = msg;
                output_q->push(std::move(entry));
            });
        }
    }
#endif

    EINSUMS_LOG_INFO("logging submodule has been initialized");
    EINSUMS_LOG_INFO("log level: {} (0=TRACE,1=DEBUG,2=INFO,3=WARN,4=ERROR,5=CRITICAL)", global_config.get_int("log-level"));
    // EINSUMS_LOG_DEBUG("test debug");
    // EINSUMS_LOG_TRACE("test trace");
    // EINSUMS_LOG_INFO("test info");
    // EINSUMS_LOG_WARN("test warn");
    // EINSUMS_LOG_ERROR("test error");
    // EINSUMS_LOG_CRITICAL("test critical");
}

} // namespace einsums::detail