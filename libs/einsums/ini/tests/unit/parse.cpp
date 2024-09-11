//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/ini/ini.hpp>

#include <einsums/testing.hpp>

TEST_CASE("ini") {
    std::vector<std::string> config = {"[system]", "pid=42", "[einsums.stacks]", "small_stack_size=64"};
    einsums::detail::section sec;
    sec.parse("<static defaults>", config, false, false, false);

    REQUIRE(sec.has_section("system"));
    REQUIRE(sec.has_section("einsums.stacks"));
    REQUIRE_FALSE(sec.has_section("einsums.thread_queue"));
}
