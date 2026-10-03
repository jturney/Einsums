//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Logging/Options.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime/Detail/InitLogging.hpp>
#include <Einsums/Runtime/RuntimeConfiguration.hpp>

#if defined(EINSUMS_HAVE_PROFILER)
#    include <Einsums/Profile/Profile.hpp>
#    include <Einsums/Profile/SpdlogSink.hpp>
#endif

#include <fmt/format.h>

#include <chrono>
#include <ctime>
#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>
#include <string>

EINSUMS_NAMESPACE_BEGIN(detail)

namespace {
// Formats "tid/----"; the placeholder could become the OS thread description.
void spdlog_format_thread_id(int pid, spdlog::details::log_msg const &, std::tm const &, spdlog::memory_buf_t &dest) {
    dest.append(fmt::format("{}/----", pid));
}
} // namespace

struct ThreadIdFormatterFlag : spdlog::custom_flag_formatter {
    void format(spdlog::details::log_msg const &msg, std::tm const &tm_time, spdlog::memory_buf_t &dest) override {
        spdlog_format_thread_id(current_process_id(), msg, tm_time, dest);
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

// Formats "hostname" (eventually "hostname/rank").
struct HostnameFormatterFlag : spdlog::custom_flag_formatter {
    void format(spdlog::details::log_msg const & /*msg*/, std::tm const & /*tm_time*/, spdlog::memory_buf_t &dest) override {
        dest.append(std::string("localhost"));
    }

    [[nodiscard]] std::unique_ptr<custom_flag_formatter> clone() const override {
        return spdlog::details::make_unique<HostnameFormatterFlag>();
    }
};

void init_logging(RuntimeConfiguration & /*config*/) {
    // Set log destination
    auto &sinks = get_einsums_logger().sinks();
    sinks.clear();
    sinks.push_back(get_spdlog_sink(config::get(option::LogDestination)));

    // Set log pattern
    auto formatter = std::make_unique<spdlog::pattern_formatter>();
    formatter->add_flag<ThreadIdFormatterFlag>('k');
    formatter->add_flag<ParentThreadIdFormatterFlag>('q');
    formatter->add_flag<HostnameFormatterFlag>('j');
    formatter->set_pattern(config::get(option::LogFormat));
    get_einsums_logger().set_formatter(std::move(formatter));

    // Set log level
    get_einsums_logger().set_level(static_cast<spdlog::level::level_enum>(config::get(option::LogLevel)));

#if defined(EINSUMS_HAVE_PROFILER)
    // Log messages and println output reach the profiler's viewers whenever a server runs, including
    // one started after this point, so these are installed whether or not a server exists yet.
    {
        auto profiler_sink = std::make_shared<profile::SpdlogSink>();
        profiler_sink->set_level(spdlog::level::trace);
        sinks.push_back(profiler_sink);

        einsums::print::set_output_sink([](std::string const &msg) { waggle::output(msg); });

        // The profiler's own messages go to this logger, under its run-time level, rather than to
        // stderr. The logger object, not EINSUMS_LOG_*, which the build's level can compile out.
        waggle::set_diagnostic_handler([](waggle::DiagnosticLevel level, std::string_view message) {
            auto const spd_level = [level] {
                switch (level) {
                case waggle::DiagnosticLevel::Debug:
                    return spdlog::level::debug;
                case waggle::DiagnosticLevel::Info:
                    return spdlog::level::info;
                case waggle::DiagnosticLevel::Warning:
                    return spdlog::level::warn;
                case waggle::DiagnosticLevel::Error:
                    break;
                }
                return spdlog::level::err;
            }();
            get_einsums_logger().log(spd_level, "profiler: {}", message);
        });
    }
#endif

    EINSUMS_LOG_INFO("logging submodule has been initialized");
    EINSUMS_LOG_INFO("log level: {} (0=TRACE,1=DEBUG,2=INFO,3=WARN,4=ERROR,5=CRITICAL)", config::get(option::LogLevel));
}

EINSUMS_NAMESPACE_END(detail)