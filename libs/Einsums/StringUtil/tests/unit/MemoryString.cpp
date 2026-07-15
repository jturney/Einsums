//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/StringUtil/MemoryString.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("Memory specifications") {
    using namespace einsums;
    using namespace einsums::string_util;

    SECTION("Good strings") {
        REQUIRE(memory_string("1") == 1);
        REQUIRE(memory_string("2B") == 2);
        REQUIRE(memory_string("3M") == 3 * 1024 * 1024);
        REQUIRE(memory_string("4GB") == 4ULL * 1024 * 1024 * 1024);
        REQUIRE(memory_string("5To") == 5ULL * 1024 * 1024 * 1024 * 1024);
        REQUIRE(memory_string("6kw") == 6 * 1024 / sizeof(size_t));
        REQUIRE(memory_string("7.8Mb") == 8178892);
    }

    SECTION("Weird strings") {
        REQUIRE(memory_string("1.234     o") == 1);
        REQUIRE(memory_string("  1.234   g     w    ") == 1324997410 / sizeof(size_t));
    }

    SECTION("Bad strings") {
        REQUIRE_THROWS(memory_string("-1"));
        REQUIRE_THROWS(memory_string("2.3.4"));
        REQUIRE_THROWS(memory_string("5MMMM"));
        REQUIRE_THROWS(memory_string("6 m g k b"));
        REQUIRE_THROWS(memory_string("7 bbbb"));
        REQUIRE_THROWS(memory_string("8 o1234"));
        REQUIRE_THROWS(memory_string("9.,.,.,"));
    }
}