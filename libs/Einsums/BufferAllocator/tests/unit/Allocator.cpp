//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/BufferAllocator/ModuleVars.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Allocations", "[memory]", double, void, std::complex<double>) {
    using namespace einsums;

    {
        auto &config = GlobalConfigMap::get_singleton();
        auto  lock   = std::lock_guard(config);

        config.get_string_map()->get_value()["buffer-size"] = "32MB"; // Set to a reasonable number of bytes.
    }

    SECTION("Big allocations") {
        BufferAllocator<TestType> alloc;

        auto *ptr = alloc.allocate(0);

        REQUIRE(ptr == nullptr);

        REQUIRE_NOTHROW(ptr = alloc.allocate(1));

        REQUIRE(ptr != nullptr);

        REQUIRE_NOTHROW(alloc.deallocate(ptr, 1));

        REQUIRE_NOTHROW(ptr = alloc.allocate(101));

        REQUIRE(ptr != nullptr);

        REQUIRE_NOTHROW(alloc.deallocate(ptr, 101));

        // Try to allocate some absurd amount of memory.
        REQUIRE_THROWS(ptr = alloc.allocate(4194304000000 / alloc.type_size + 1));
    }

    SECTION("Small allocations") {
        auto &config = GlobalConfigMap::get_singleton();

        auto hold_str = config.get_string("buffer-size");

        {
            auto lock = std::lock_guard(config);

            config.get_string_map()->get_value()["buffer-size"] = "1024"; // Set to some small number of bytes.
        }

        BufferAllocator<TestType> alloc;

        auto *ptr = alloc.allocate(0);

        REQUIRE(ptr == nullptr);

        REQUIRE_NOTHROW(ptr = alloc.allocate(1));

        REQUIRE(ptr != nullptr);

        REQUIRE_NOTHROW(alloc.deallocate(ptr, 1));

        REQUIRE_THROWS(ptr = alloc.allocate(1025));

        {
            auto lock = std::lock_guard(config);

            config.get_string_map()->get_value()["buffer-size"] = hold_str; // Set to some small number of bytes.
        }
    }
}

TEMPLATE_TEST_CASE("Vector", "[memory]", int, double, std::complex<double>) {
    using namespace einsums;

    auto &config = GlobalConfigMap::get_singleton();

    auto hold_str = config.get_string("buffer-size");

    {
        auto lock = std::lock_guard(config);

        config.get_string_map()->get_value()["buffer-size"] = "1024"; // Set to some small number of bytes.
    }

    // First, check for the new value of the buffer size.
    auto &vars = detail::Einsums_BufferAllocator_vars::get_singleton();
    REQUIRE(vars.get_max_size() == 1024);

    size_t start_size = vars.get_available();

    // Allocate stuff.
    std::vector<TestType, BufferAllocator<TestType>> vec1;

    REQUIRE_NOTHROW(vec1.resize(1));
    REQUIRE(vec1.size() == 1);
    REQUIRE(vars.get_available() == start_size - 1 * sizeof(TestType));
    REQUIRE_NOTHROW(vec1.at(0) = TestType{0});
    REQUIRE(vec1[0] == TestType{0});

    REQUIRE_NOTHROW(vec1.resize(20));
    REQUIRE(vec1.size() == 20);
    REQUIRE(vars.get_available() == start_size - 20 * sizeof(TestType));
    REQUIRE_NOTHROW(vec1.at(19) = TestType{0});
    REQUIRE(vec1[19] == TestType{0});

    vec1.clear();
    vec1.shrink_to_fit();

    REQUIRE(vars.get_available() == start_size);

    REQUIRE_THROWS(vec1.resize(start_size / sizeof(int) + 1));
}