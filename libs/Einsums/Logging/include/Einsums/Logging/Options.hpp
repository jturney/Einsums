//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

#include <spdlog/common.h>

/*
 * The logging subsystem's options, declared where the subsystem is configured.
 */

EINSUMS_NAMESPACE_BEGIN(option)

/// spdlog's numeric level: 0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR.
inline constinit cl::ConfigOption<std::int64_t> LogLevel = cl::config_opt<std::int64_t>("einsums:log:level", "Log level", "Logging",
#if defined(EINSUMS_DEBUG)
                                                                                        SPDLOG_LEVEL_DEBUG,
#else
                                                                                        SPDLOG_LEVEL_ERROR,
#endif
                                                                                        "LogLevel", cl::RangeBetween(0, 4));

/// Where log records go: `cerr`, `cout`, or a file name.
inline constinit cl::ConfigOption<std::string> LogDestination =
    cl::config_opt<std::string>("einsums:log:destination", "Log destination", "Logging", "cerr");

/// The spdlog pattern each record is rendered with.
inline constinit cl::ConfigOption<std::string> LogFormat =
    cl::config_opt<std::string>("einsums:log:format", "Log format", "Logging", "[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%-8l%$] [%s:%#/%!] %v");

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give the logging options their command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_Logging_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_Logging = register_Einsums_Logging_options();
}

EINSUMS_NAMESPACE_END()
