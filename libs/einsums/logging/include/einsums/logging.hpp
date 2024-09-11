//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <memory>
#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>

namespace einsums::detail {

#define EINSUMS_DETAIL_DECLARE_SPDLOG(name) spdlog::logger &get_##name##_logger() noexcept;

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

#define EINSUMS_DETAIL_SPDLOG(name, loglevel, ...)                                                                                         \
    if (::einsums::detail::get_##name##_logger().level() <= spdlog::level::loglevel) {                                                     \
        ::einsums::detail::get_##name##_logger().log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::loglevel,     \
                                                     __VA_ARGS__);                                                                         \
    }

#define EINSUMS_DETAIL_SPDLOG_ENABLED(name, loglevel) (::einsums::detail::get_##name##_logger().level() <= spdlog::level::loglevel)
} // namespace einsums::detail

#define EINSUMS_LOG(loglevel, ...)    EINSUMS_DETAIL_SPDLOG(einsums, loglevel, __VA_ARGS__)
#define EINSUMS_LOG_ENABLED(loglevel) EINSUMS_DETAIL_SPDLOG_ENABLED(einsums, loglevel)

namespace einsums::detail {
EINSUMS_EXPORT spdlog::level::level_enum get_spdlog_level(std::string const &env);
EINSUMS_EXPORT std::shared_ptr<spdlog::sinks::sink> get_spdlog_sink(std::string const &env);
EINSUMS_EXPORT                                      EINSUMS_DETAIL_DECLARE_SPDLOG(einsums)
} // namespace einsums::detail
