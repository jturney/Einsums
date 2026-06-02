//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Logging/Defines.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>

namespace einsums::detail {

/**
 * @brief Create the prototype of a logger.
 *
 * @param name The name of the logger to create.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_DETAIL_DECLARE_SPDLOG(name) spdlog::logger &get_##name##_logger() noexcept;

/**
 * @brief Create the definition of a logger.
 *
 * @param name The name of the logger to create.
 * @param loglevel The log level for the sink.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_DETAIL_DEFINE_SPDLOG(name, loglevel)                                                                                       \
    spdlog::logger &get_##name##_logger() noexcept {                                                                                       \
        static auto logger = []() {                                                                                                        \
            auto logger = std::make_shared<spdlog::logger>(#name, std::make_shared<spdlog::sinks::stderr_color_sink_mt>());                \
            logger->set_level(spdlog::level::loglevel);                                                                                    \
            return logger;                                                                                                                 \
        }();                                                                                                                               \
        static auto &logger_ref = *logger;                                                                                                 \
        return logger_ref;                                                                                                                 \
    }

/**
 * @brief Implementation detail for the log macros.
 *
 * This macro can also take a format string and the arguments to support it.
 *
 * @param name The name of the logger.
 * @param loglevel The level of the message.
 * @param ... The format string and arguments to support it.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_DETAIL_SPDLOG(name, loglevel, ...)                                                                                         \
    if (::einsums::detail::get_##name##_logger().level() <= loglevel) {                                                                    \
        ::einsums::detail::get_##name##_logger().log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION},                              \
                                                     static_cast<::spdlog::level::level_enum>(loglevel), __VA_ARGS__);                     \
    }

/**
 * Checks to see if the logger can handle the given log level.
 *
 * @param name The name of the logger.
 * @param loglevel The leve to check.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_DETAIL_SPDLOG_ENABLED(name, loglevel) (::einsums::detail::get_##name##_logger().level() <= loglevel)

/**
 * Log a message at the given level.
 *
 * @param loglevel The level to log at.
 * @param ... The format string and arguments.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_LOG(loglevel, ...) EINSUMS_DETAIL_SPDLOG(einsums, loglevel, __VA_ARGS__)

/**
 * Determine if Einsums has the given log level enabled.
 *
 * @param loglevel The level to test.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_LOG_ENABLED(loglevel) EINSUMS_DETAIL_SPDLOG_ENABLED(einsums, loglevel)

// Only EINSUMS_LOG_TRACE and EINSUMS_LOG_DEBUG can be disabled during compile time.
/**
 * @def EINSUMS_LOG_TRACE
 *
 * Log a tracing message. These can be disabled at compile time by compiling in the Release configuration. Trace messages are the lowest
 * priority and are used for intensive debugging. It can take a format string and the arguments for that format string.
 *
 * @param ... The format string and arguments.
 *
 * @versionadded{1.0.0}
 */
#if EINSUMS_ACTIVE_LOG_LEVEL <= 0
#    define EINSUMS_LOG_TRACE(...) EINSUMS_LOG(SPDLOG_LEVEL_TRACE, __VA_ARGS__)
#else
#    define EINSUMS_LOG_TRACE(...)
#endif

/**
 * @def EINSUMS_LOG_DEBUG
 *
 * Log a debugging message. These are messages that may help a maintainer figure out what is happening in the code. They can be disabled at
 * compile time by compiling in the Release configuration. It can take a format string and the arguments for that format string.
 *
 * @param ... The format string and arguments.
 *
 * @versionadded{1.0.0}
 */
#if EINSUMS_ACTIVE_LOG_LEVEL <= 1
#    define EINSUMS_LOG_DEBUG(...) EINSUMS_LOG(SPDLOG_LEVEL_DEBUG, __VA_ARGS__)
#else
#    define EINSUMS_LOG_DEBUG(...)
#endif

// These logging macros cannot be disabled at compile-time, but they can be disabled at runtime
/**
 * @def EINSUMS_LOG_INFO
 *
 * Log an informational message. These indicate things that are not errors or bad behavior, but rather things that tell you about the
 * program you are running, such as configuration values. It can take a format string and the arguments for that format string.
 *
 * @param ... The format string and arguments.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_LOG_INFO(...) EINSUMS_LOG(SPDLOG_LEVEL_INFO, __VA_ARGS__)

/**
 * @def EINSUMS_LOG_WARN
 *
 * Log a warning message. These are messages logged when a recoverable issue occurs in the code. It can take a format
 * string and the arguments for that format string.
 *
 * @param ... The format string and arguments.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_LOG_WARN(...) EINSUMS_LOG(SPDLOG_LEVEL_WARN, __VA_ARGS__)

/**
 * @def EINSUMS_LOG_ERROR
 *
 * Log an error message. These are messages logged when an error occurs in the code. These are often accompanied by an exception. It can
 * take a format string and the arguments for that format string.
 *
 * @param ... The format string and arguments.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_LOG_ERROR(...) EINSUMS_LOG(SPDLOG_LEVEL_ERROR, __VA_ARGS__)

/**
 * @def EINSUMS_LOG_CRITICAL
 *
 * Log a critical error message. These are messages logged when an unrecoverable error occurs in the code. It can take a format
 * string and the arguments for that format string.
 *
 * @param ... The format string and arguments.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_LOG_CRITICAL(...) EINSUMS_LOG(SPDLOG_LEVEL_CRITICAL, __VA_ARGS__)

/**
 * @brief Get the log level corresponding to the given numerical string.
 *
 * @param[in] env The numeric string representing the log level.
 *
 * @return The enum value corresponding to the number represented by the input string.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT spdlog::level::level_enum get_spdlog_level(std::string const &env);

/**
 * @brief Get the sink corresponding to the input string.
 *
 * @param[in] env The string representing the sink. Can be "cerr", "cout", or a file name.
 *
 * @return The sink corresponding to the input string.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT std::shared_ptr<spdlog::sinks::sink> get_spdlog_sink(std::string const &env);

EINSUMS_EXPORT EINSUMS_DETAIL_DECLARE_SPDLOG(einsums)

} // namespace einsums::detail

namespace einsums {

/// Set the runtime log level on the einsums logger. Values mirror
/// spdlog: 0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR, 5=CRITICAL, 6=OFF.
/// Out-of-range values are clamped by spdlog. Compile-time disablement
/// (``EINSUMS_ACTIVE_LOG_LEVEL``) takes precedence — runtime
/// ``set_log_level(0)`` cannot re-enable TRACE in a Release build.
EINSUMS_PYBIND_EXPOSE EINSUMS_EXPORT void set_log_level(int level);

/// Return the current runtime log level on the einsums logger.
EINSUMS_PYBIND_EXPOSE EINSUMS_EXPORT int get_log_level();

} // namespace einsums