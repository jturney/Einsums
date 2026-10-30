//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Logging.hpp>

#include <cstring>
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <typeinfo>

#include <Einsums/Testing.hpp>

using namespace einsums;

namespace {
// Whether the sink is a Sink, by the type's name rather than by dynamic_cast. A shared spdlog
// exports its sinks' type_info while this test, compiled with hidden visibility, emits a private
// copy of its own; libc++ then compares the two by address, and the cast fails on a sink of
// exactly the right type.
template <typename Sink>
bool is_sink(std::shared_ptr<spdlog::sinks::sink> const &sink) {
    if (sink == nullptr) {
        return false;
    }
    spdlog::sinks::sink const &object = *sink;
    return std::strcmp(typeid(object).name(), typeid(Sink).name()) == 0;
}
} // namespace

// An empty destination announced that it was using cerr, then opened a file with an empty name,
// which spdlog refuses with an exception.
TEST_CASE("An empty log destination writes to standard error", "[Logging]") {
    std::shared_ptr<spdlog::sinks::sink> sink;
    REQUIRE_NOTHROW(sink = detail::get_spdlog_sink(""));
    CHECK(is_sink<spdlog::sinks::stderr_color_sink_mt>(sink));
}

TEST_CASE("The named destinations map to the console streams", "[Logging]") {
    CHECK(is_sink<spdlog::sinks::stderr_color_sink_mt>(detail::get_spdlog_sink("cerr")));
    CHECK(is_sink<spdlog::sinks::stdout_color_sink_mt>(detail::get_spdlog_sink("cout")));
}

TEST_CASE("Pass narration is not filtered by the library's log level", "[Logging]") {
    auto const saved = get_log_level();
    set_log_level(4); // ERROR
    CHECK(detail::get_passes_logger().should_log(spdlog::level::info));
    set_log_level(saved);
}
