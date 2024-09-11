//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/init_runtime/detail/init_logging.hpp>
#include <einsums/logging.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>
#include <string>

namespace einsums::detail {

struct log_settings {
    std::string level;
    std::string dest;
    std::string format;
};

static log_settings get_log_settings(section const &ini, char const *sec) {
    log_settings result;
    if (ini.has_section(sec)) {
        section const *logini = ini.get_section(sec);
        EINSUMS_ASSERT(logini != nullptr);

        result.level  = logini->get_entry("level", "");
        result.dest   = logini->get_entry("destination", "");
        result.format = logini->get_entry("format", "");
    }
    return result;
}

void init_logging(util::runtime_configuration &ini) {
    auto settings = get_log_settings(ini, "einsums.log");

    // Set log destination
    auto &sinks = get_einsums_logger().sinks();
    sinks.clear();
    sinks.push_back(get_spdlog_sink(settings.dest));

    // Set log pattern
    auto formatter = std::make_unique<spdlog::pattern_formatter>();

    // add custom formatters here

    formatter->set_pattern(settings.format);
    get_einsums_logger().set_formatter(std::move(formatter));

    // Set log level
    get_einsums_logger().set_level(get_spdlog_level(settings.level));
}

} // namespace einsums::detail