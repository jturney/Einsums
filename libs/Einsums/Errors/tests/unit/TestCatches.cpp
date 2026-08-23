//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <Einsums/Testing.hpp>

[[noreturn]] static void thrower() {
    EINSUMS_THROW_EXCEPTION(einsums::DimensionError, "Test error.");
}

[[noreturn]] static void thrower2() {
    throw einsums::DimensionError("Test error2.");
}

[[noreturn]] static void thrower3() {
    throw std::logic_error("Logic Error test");
}

TEST_CASE("Test catching 1", "[error]") {
    INFO("Starting test.");
    try {
        INFO("Throwing...");
        thrower();
    } catch (einsums::DimensionError const &e) {
        INFO("Caught exception!");
        REQUIRE(true);
    } catch (...) {
        INFO("Caught the exception, but not in the correct exception handler!");
    }
    INFO("Finished handling exception.");

    REQUIRE_THROWS_AS(thrower(), einsums::DimensionError);
}

TEST_CASE("Test catching 2", "[error]") {
    INFO("Starting test.");
    try {
        INFO("Throwing...");
        thrower2();
    } catch (einsums::DimensionError const &e) {
        INFO("Caught exception!");
        REQUIRE(true);
    } catch (...) {
        INFO("Caught the exception, but not in the correct exception handler!");
    }
    INFO("Finished handling exception.");

    REQUIRE_THROWS_AS(thrower2(), einsums::DimensionError);
}

TEST_CASE("Test catching 3", "[error]") {
    INFO("Starting test.");
    try {
        INFO("Throwing...");
        thrower3();
    } catch (std::logic_error const &e) {
        INFO("Caught exception!");
        REQUIRE(true);
    } catch (...) {
        INFO("Caught the exception, but not in the correct exception handler!");
    }
    INFO("Finished handling exception.");

    REQUIRE_THROWS_AS(thrower3(), std::logic_error);
}

TEST_CASE("Test catching 4", "[error]") {
    INFO("Starting test.");

    std::vector<int> vec(0);

    try {
        INFO("Throwing...");

        auto x = vec.at(1);
    } catch (std::out_of_range const &e) {
        INFO("Caught exception!");
        REQUIRE(true);
    } catch (...) {
        INFO("Caught the exception but not in the correct handler!");
    }

    int x;

    REQUIRE_THROWS_AS((x = vec.at(1)), std::out_of_range);
}