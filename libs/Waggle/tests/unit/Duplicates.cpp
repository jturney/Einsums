//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// A second copy of the collector, loaded beside the first as a Python extension that bundles its
// own would be, finds the first, switches itself off, and tells it.

#include <Waggle/Config.hpp>

#include <Waggle/Waggle.hpp>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>

#include "Profiler.hpp"

#if defined(__APPLE__) || defined(__linux__)
#    include <dlfcn.h>

TEST_CASE("A second copy of the collector switches itself off and is reported", "[profiler][duplicates]") {
    // This copy is active: the test links it.
    waggle::set_enabled(true);
    REQUIRE(waggle::Profiler::instance().duplicates().empty());

    // As Python loads an extension: privately, so its symbols stay out of the global scope.
    void *second = dlopen(WAGGLE_SECOND_COPY_PATH, RTLD_NOW | RTLD_LOCAL);
    REQUIRE(second != nullptr);

    // The second copy is off: its switch reads off, it hands out no ids and takes no snapshots.
    auto const enabled_flag  = reinterpret_cast<int32_t const *(*)()>(dlsym(second, "waggle_enabled_flag"));
    auto const register_site = reinterpret_cast<uint32_t (*)(char const *, size_t, char const *, int, char const *, uint32_t)>(
        dlsym(second, "waggle_register_site"));
    auto const snapshot_take = reinterpret_cast<waggle_snapshot *(*)(uint32_t)>(dlsym(second, "waggle_snapshot_take"));
    auto const set_enabled   = reinterpret_cast<void (*)(int)>(dlsym(second, "waggle_set_enabled"));
    REQUIRE(enabled_flag != nullptr);
    REQUIRE(register_site != nullptr);
    REQUIRE(snapshot_take != nullptr);
    REQUIRE(set_enabled != nullptr);
    CHECK(enabled_flag() != waggle_enabled_flag());
    set_enabled(1); // turning it on changes nothing
    CHECK(*enabled_flag() == 0);
    CHECK(register_site("duplicates: second copy", 23, __FILE__, __LINE__, "test", 0) == 0);
    CHECK(snapshot_take(0) == nullptr);

    // The first copy knows, and names the second.
    auto const duplicates = waggle::Profiler::instance().duplicates();
    REQUIRE(duplicates.size() == 1);
    CHECK(duplicates[0].path.find("waggle_second_copy") != std::string::npos);
    CHECK(duplicates[0].abi_major == WAGGLE_ABI_MAJOR);
    CHECK(duplicates[0].abi_minor == WAGGLE_ABI_MINOR);

    // Its report says what is missing.
    std::ostringstream report;
    waggle::Profiler::instance().print(false, report);
    CHECK(report.str().find("Not recorded") != std::string::npos);
    CHECK(report.str().find("waggle_second_copy") != std::string::npos);

    // And keeps recording.
    std::thread([] { WAGGLE_ZONE("duplicates: first copy still records"); }).join();
    CHECK(waggle::Snapshot::take().find("duplicates: first copy still records"));

    dlclose(second);
}

#endif
