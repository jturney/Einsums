//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Logging.hpp>

#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <Einsums/Testing.hpp>

using namespace einsums;

// An empty destination announced that it was using cerr, then opened a file with an empty name,
// which spdlog refuses with an exception.
TEST_CASE("An empty log destination writes to standard error", "[Logging]") {
    std::shared_ptr<spdlog::sinks::sink> sink;
    REQUIRE_NOTHROW(sink = detail::get_spdlog_sink(""));
    CHECK(std::dynamic_pointer_cast<spdlog::sinks::stderr_color_sink_mt>(sink) != nullptr);
}

TEST_CASE("The named destinations map to the console streams", "[Logging]") {
    CHECK(std::dynamic_pointer_cast<spdlog::sinks::stderr_color_sink_mt>(detail::get_spdlog_sink("cerr")) != nullptr);
    CHECK(std::dynamic_pointer_cast<spdlog::sinks::stdout_color_sink_mt>(detail::get_spdlog_sink("cout")) != nullptr);
}

TEST_CASE("Pass narration is not filtered by the library's log level", "[Logging]") {
    auto const saved = get_log_level();
    set_log_level(4); // ERROR
    CHECK(detail::get_passes_logger().should_log(spdlog::level::info));
    set_log_level(saved);
}
