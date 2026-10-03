//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Profile/Profile.hpp>

#include <fmt/format.h>

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <sstream>
#include <thread>

using namespace waggle;

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

TEST_CASE("WAGGLE_ZONE macro works", "[profiler][consumer]") {
    {
        WAGGLE_ZONE("labeled_macro_{}", 42);
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

// ═══════════════════════════════════════════════════════════════════════════
// Reconstruction under dropped events
// ═══════════════════════════════════════════════════════════════════════════

/// Deepest chain of nodes carrying @p name anywhere in the forest.
///
/// One is correct: a zone pushed and popped from the same place is one call
/// path however many times it runs. More means the consumer nested a zone under
/// itself, which is what a lost Pop used to make it do - one extra level per
/// dropped event, without bound.
static size_t deepest_run_of(AggNode const &node, std::string const &name, size_t run = 0) {
    size_t const here = (node.name == name) ? run + 1 : 0;
    size_t       best = here;
    for (auto const &c : node.children) {
        best = std::max(best, deepest_run_of(*c.second, name, here));
    }
    return best;
}

TEST_CASE("Profiler - a burst that overruns the ring buffer does not deepen the tree", "[profiler][consumer]") {
    auto &prof = Profiler::instance();
    if (!prof.enabled()) {
        SKIP("the profiler is disabled, so no events are recorded at all");
    }

    // Balanced push/pop pairs, more of them than the ring buffer holds and with
    // nothing between them to let the consumer catch up. Some of these events
    // WILL be dropped, and which ones is up to the drain: the point is that a
    // dropped Push or Pop costs its own zone's timing and changes nothing about
    // the shape of the tree.
    auto const     site_id        = prof.register_site("ring_burst_zone", "burst.cpp", 1, "burst");
    uint64_t const dropped_before = prof.consumer()->dropped_count();

    constexpr int kPairs = 200'000; // ring buffer holds 65'536 events
    for (int i = 0; i < kPairs; i++) {
        prof.push_interned(site_id, 0);
        prof.pop();
    }

    prof.flush();

    auto        lock       = prof.consumer()->lock_shared();
    auto const &thread_map = prof.consumer()->thread_data();

    size_t deepest = 0;
    for (auto const &tkv : thread_map) {
        deepest = std::max(deepest, deepest_run_of(tkv.second.root, "ring_burst_zone"));
    }
    REQUIRE(deepest == 1);

    if (prof.consumer()->dropped_count() == dropped_before) {
        WARN("the burst kept up with the consumer, so this run did not exercise reconstruction under drops");
    }
}

// A zone name built at run time, such as a graph's, made a new node under its parent for every
// distinct value, and every zone beneath it a new node too. A fuzzer that named each trial's graph
// afresh grew the tree by ~160 KB per trial until a scaled nightly run exhausted the runner. Past
// option::ProfileMaxDistinctChildren (256 by default) new names fold into the parent's "(other)".
TEST_CASE("Profiler - names past the distinct-children cap fold into one node", "[profiler][consumer]") {
    auto &prof = Profiler::instance();
    if (!prof.enabled()) {
        SKIP("the profiler is disabled, so no events are recorded at all");
    }

    constexpr int kCap   = 256;
    constexpr int kNames = 1000;
    auto const    zone   = [&prof](std::string const &name) {
        prof.push(name, "cap.cpp", 1, "cap");
        prof.push("distinct_cap_leaf", "cap.cpp", 2, "cap");
        prof.pop();
        prof.pop();
    };

    prof.push("distinct_cap_parent", "cap.cpp", 0, "cap");
    for (int i = 0; i < kNames; i++) {
        zone("distinct_cap_child_" + std::to_string(i));
    }
    // After the cap: a name that came first still has its own node, and a folded name folds again
    // without counting as another distinct name.
    zone("distinct_cap_child_0");
    zone("distinct_cap_child_" + std::to_string(kNames - 1));
    prof.pop();
    prof.flush();

    auto        lock       = prof.consumer()->lock_shared();
    auto const &thread_map = prof.consumer()->thread_data();

    AggNode const *parent = find_node_any_thread(thread_map, "distinct_cap_parent");
    REQUIRE(parent != nullptr);
    REQUIRE(parent->children.size() == kCap + 1);

    AggNode const *kept = find_node(*parent, "distinct_cap_child_0");
    REQUIRE(kept != nullptr);
    CHECK(kept->call_count == 2);
    CHECK(find_node(*parent, "distinct_cap_child_" + std::to_string(kCap - 1)) != nullptr);
    CHECK(find_node(*parent, "distinct_cap_child_" + std::to_string(kCap)) == nullptr);

    AggNode const *other = find_node(*parent, "(other)");
    REQUIRE(other != nullptr);
    CHECK(other->call_count == kNames - kCap + 1);
    auto const distinct = other->annotations.find("distinct");
    REQUIRE(distinct != other->annotations.end());
    CHECK(distinct->second == std::to_string(kNames - kCap));

    // The folded zones' own children merge under "(other)" rather than being dropped.
    REQUIRE(other->children.size() == 1);
    AggNode const *leaf = find_node(*other, "distinct_cap_leaf");
    REQUIRE(leaf != nullptr);
    CHECK(leaf->call_count == kNames - kCap + 1);
}

TEST_CASE("Profiler - the timeline keeps the latest zones, oldest first", "[profiler][consumer]") {
    // The timeline used to be a vector trimmed with erase(begin()), so once full every closed zone
    // shifted a thousand events down by one. It is now a ring; read back, it must still be the most
    // recent zones in the order they closed.
    auto &prof = Profiler::instance();
    prof.flush();
    size_t const total = Consumer::kMaxTimelineEvents + 250;
    // Under a parent of their own: the tree caps the distinct names under one parent and folds the
    // rest into "(other)", so this many names at the root would fold every later test's zones. The
    // timeline records each zone's own name either way.
    prof.push("timeline test parent");
    for (size_t i = 0; i < total; ++i) {
        auto const name = "timeline zone " + std::to_string(i);
        prof.push(name);
        prof.pop();
    }
    prof.pop();
    prof.flush();

    auto       lock   = prof.consumer()->lock_shared();
    auto const events = prof.consumer()->timeline_events();
    REQUIRE(events.size() == Consumer::kMaxTimelineEvents);
    // The parent closes last, so it is the newest record.
    CHECK(events.back().name == "timeline test parent");
    CHECK(events[events.size() - 2].name == "timeline zone " + std::to_string(total - 1));
    CHECK(events.front().name == "timeline zone " + std::to_string(total + 1 - Consumer::kMaxTimelineEvents));
    for (size_t k = 1; k < events.size(); ++k) {
        CHECK(events[k].end_ms >= events[k - 1].end_ms);
    }
}

TEST_CASE("Profiler - no counter rows without a counter backend", "[profiler][consumer]") {
    // Every closed zone merged four hardware counters into string-keyed maps, all zero when no
    // backend is active, and every node of the report carried four rows of zeros.
    if (get_counter_backend().available()) {
        SKIP("a hardware counter backend is active on this machine");
    }
    auto &prof = Profiler::instance();
    {
        WAGGLE_ZONE("counterless zone");
    }
    prof.flush();
    auto        lock = prof.consumer()->lock_shared();
    auto const *node = find_node_any_thread(prof.consumer()->thread_data(), "counterless zone");
    REQUIRE(node != nullptr);
    CHECK(node->counters_total.empty());
}

TEST_CASE("Profiler - a zone opened in the library nests under the caller's zone", "[profiler][consumer]") {
    // The per-thread channel was reached through an inline function holding a thread_local, and
    // -fvisibility-inlines-hidden gives every shared object its own copy: this executable and the
    // library each had a channel for the thread, with its own ring and depth, and the library's
    // zones landed at the root instead of under the zone that called them. dgemv opens its zone in
    // the library, not in anything inlined here.
    auto &prof = Profiler::instance();
    {
        WAGGLE_ZONE("caller zone across libraries");
        std::array<double, 4> const a{1.0, 2.0, 3.0, 4.0};
        std::array<double, 2> const x{1.0, 1.0};
        std::array<double, 2>       y{};
        einsums::blas::vendor::dgemv('n', 2, 2, 1.0, a.data(), 2, x.data(), 1, 0.0, y.data(), 1);
    }
    prof.flush();

    auto        lock   = prof.consumer()->lock_shared();
    auto const *caller = find_node_any_thread(prof.consumer()->thread_data(), "caller zone across libraries");
    REQUIRE(caller != nullptr);
    bool nested = false;
    for (auto const &child : caller->children) {
        nested = nested || child.second->name == "dgemv";
    }
    CHECK(nested);
}

// The report's per-thread header summed only the outermost zones' exclusive time, so a thread whose
// time was all in nested zones reported almost nothing: 0.000 ms over a worker that ran 14 ms. The
// thread is named before its first zone, which the automatic name used to overwrite.
TEST_CASE("The report's thread total includes nested zones", "[profiler][consumer][report]") {
    auto &prof = Profiler::instance();
    prof.flush();

    uint32_t    tid = 0;
    std::thread worker([&] {
        tid = Profiler::current_thread_id();
        prof.set_thread_name("report-total-test");
        prof.push("report_total_outer");
        prof.push("report_total_inner");
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        prof.pop();
        prof.pop();
    });
    worker.join();
    prof.flush();

    double expected_ms        = 0.0;
    double outer_exclusive_ms = 0.0;
    {
        auto lock = prof.consumer()->lock_shared();
        auto it   = prof.consumer()->thread_data().find(tid);
        REQUIRE(it != prof.consumer()->thread_data().end());
        auto const *outer = find_node(it->second.root, "report_total_outer");
        REQUIRE(outer != nullptr);
        expected_ms        = std::chrono::duration<double, std::milli>(inclusive_time(it->second.root)).count();
        outer_exclusive_ms = std::chrono::duration<double, std::milli>(outer->total_exclusive).count();
    }
    REQUIRE(expected_ms >= 3.0);
    REQUIRE(outer_exclusive_ms < 1.0); // what the header used to show

    std::ostringstream report;
    prof.print(false, report);
    std::string const text   = report.str();
    std::string const header = fmt::format("Thread: report-total-test ({})  (total exclusive:", tid);
    auto const        at     = text.find(header);
    REQUIRE(at != std::string::npos);
    double const shown = std::stod(text.substr(at + header.size()));
    CHECK_THAT(shown, Catch::Matchers::WithinAbs(expected_ms, 0.001));
}
