//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Profile/Profile.hpp>

#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <thread>

#if defined(EINSUMS_HAVE_PROFILER)

using namespace einsums::profile;

// Helper: recursively search the AggNode tree for a node with a given name.
static AggNode const *find_node(AggNode const &root, std::string const &name) {
    for (auto const &c : root.children) {
        if (c.second->name == name)
            return c.second.get();
        auto *found = find_node(*c.second, name);
        if (found)
            return found;
    }
    return nullptr;
}

static AggNode const *find_node_any_thread(std::unordered_map<uint32_t, ThreadState> const &thread_map, std::string const &name) {
    for (auto const &tkv : thread_map) {
        auto *found = find_node(tkv.second.root, name);
        if (found)
            return found;
    }
    return nullptr;
}

// Wait for consumer to drain events (polls up to ~200ms)
static void wait_for_drain() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

TEST_CASE("Profiler push/pop produces aggregated tree", "[profiler][consumer]") {
    auto &prof = Profiler::instance();

    prof.push("test_zone_pp", "test.cpp", 10, "test_func");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    prof.pop();

    wait_for_drain();

    auto        lock       = prof.consumer()->lock_shared();
    auto const &thread_map = prof.consumer()->thread_data();

    auto const *node = find_node_any_thread(thread_map, "test_zone_pp");
    REQUIRE(node != nullptr);
    REQUIRE(node->call_count >= 1);
    REQUIRE(node->file == "test.cpp");
    REQUIRE(node->line == 10);
    REQUIRE(node->function == "test_func");
}

TEST_CASE("Profiler nested zones produce hierarchy", "[profiler][consumer]") {
    auto &prof = Profiler::instance();

    prof.push("parent_nest", "test.cpp", 1, "test");
    prof.push("child_nest", "test.cpp", 2, "test");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    prof.pop();
    prof.pop();

    wait_for_drain();

    auto        lock       = prof.consumer()->lock_shared();
    auto const &thread_map = prof.consumer()->thread_data();

    auto const *parent = find_node_any_thread(thread_map, "parent_nest");
    REQUIRE(parent != nullptr);

    auto const *child = find_node(*parent, "child_nest");
    REQUIRE(child != nullptr);
}

TEST_CASE("ScopedZone RAII works", "[profiler][consumer]") {
    {
        ScopedZone z("scoped_raii", "test.cpp", 42, "test_func");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    wait_for_drain();

    auto        lock       = Profiler::instance().consumer()->lock_shared();
    auto const &thread_map = Profiler::instance().consumer()->thread_data();

    auto const *node = find_node_any_thread(thread_map, "scoped_raii");
    REQUIRE(node != nullptr);
    REQUIRE(node->call_count >= 1);
}

TEST_CASE("Profiler print produces output", "[profiler][consumer]") {
    auto &prof = Profiler::instance();

    prof.push("print_out_test");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    prof.pop();

    wait_for_drain();

    std::ostringstream oss;
    prof.print(false, oss);

    std::string output = oss.str();
    REQUIRE(output.find("print_out_test") != std::string::npos);
}

TEST_CASE("LabeledSection macro works", "[profiler][consumer]") {
    {
        LabeledSection("labeled_macro_{}", 42);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    wait_for_drain();

    auto        lock       = Profiler::instance().consumer()->lock_shared();
    auto const &thread_map = Profiler::instance().consumer()->thread_data();

    auto const *node = find_node_any_thread(thread_map, "labeled_macro_42");
    REQUIRE(node != nullptr);
}

#endif
