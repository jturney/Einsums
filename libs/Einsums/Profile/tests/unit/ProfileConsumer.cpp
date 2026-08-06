//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Profile/Profile.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <sstream>
#include <thread>

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

// ═══════════════════════════════════════════════════════════════════════════
// Per-sample statistics
// ═══════════════════════════════════════════════════════════════════════════
//
// Driven through AggNode directly rather than through push/pop: the durations
// that broke this are tens of seconds, and no test should spend them.

TEST_CASE("AggNode::record_exclusive - mean and variance of a small sample", "[profiler][consumer]") {
    AggNode node("stats");
    for (int64_t v : {10, 20, 30, 40}) {
        node.record_exclusive(ns{v});
    }

    REQUIRE(node.call_count == 4);
    REQUIRE(node.total_exclusive == ns{100});
    REQUIRE(node.exclusive_min == ns{10});
    REQUIRE(node.exclusive_max == ns{40});

    // Mean 25, sample variance 500/3. The integer form this replaced advanced
    // the mean by a truncating `delta / call_count` and drifted low here.
    REQUIRE_THAT(node.total_exclusive_mean, Catch::Matchers::WithinRel(25.0, 1e-12));
    REQUIRE_THAT(node.total_exclusive_M2, Catch::Matchers::WithinRel(500.0, 1e-12));
}

TEST_CASE("AggNode::record_exclusive - a multi-second spread does not overflow", "[profiler][consumer]") {
    // The nightly sanitizer leg's numbers: zones get slow enough under
    // address+undefined that one sample sits ~108 s from the mean. Squared
    // that is ~5.8e21 against an int64 ceiling of 9.2e18, which UBSan reported
    // as signed integer overflow at the Welford update.
    constexpr int64_t kBig   = 108'037'677'238; // ns, ~108 s
    constexpr int64_t kSmall = 1'000;           // ns

    AggNode node("slow_zone");
    node.record_exclusive(ns{kBig});
    node.record_exclusive(ns{kSmall});

    REQUIRE(node.call_count == 2);
    REQUIRE(node.exclusive_min == ns{kSmall});
    REQUIRE(node.exclusive_max == ns{kBig});

    double const mean = 0.5 * (static_cast<double>(kBig) + static_cast<double>(kSmall));
    REQUIRE_THAT(node.total_exclusive_mean, Catch::Matchers::WithinRel(mean, 1e-12));

    // M2 for two samples is (a-b)^2 / 2, and must stay positive and finite:
    // the overflowing form produced a wrapped, meaningless value here.
    double const diff = static_cast<double>(kBig) - static_cast<double>(kSmall);
    REQUIRE_THAT(node.total_exclusive_M2, Catch::Matchers::WithinRel(0.5 * diff * diff, 1e-12));
    REQUIRE(node.total_exclusive_M2 > 0.0);
    REQUIRE(std::isfinite(node.total_exclusive_M2));
}
