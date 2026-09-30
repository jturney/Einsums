//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Profile/Profile.hpp>

#include <fmt/ranges.h>

#include <catch2/catch_test_macros.hpp>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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

static void wait_for_drain() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

TEST_CASE("String annotations appear in tree", "[profiler][annotations]") {
    auto &prof = Profiler::instance();

    prof.push("annot_str_test");
    annotate("algorithm", "GEMM");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    prof.pop();

    wait_for_drain();

    auto        lock       = prof.consumer()->lock_shared();
    auto const &thread_map = prof.consumer()->thread_data();

    auto const *node = find_node_any_thread(thread_map, "annot_str_test");
    REQUIRE(node != nullptr);
    auto it = node->annotations.find("algorithm");
    REQUIRE(it != node->annotations.end());
    REQUIRE(it->second == "GEMM");
}

TEST_CASE("Integer annotations appear in tree", "[profiler][annotations]") {
    auto &prof = Profiler::instance();

    prof.push("annot_int_test");
    annotate("MC", int64_t(256));
    annotate("NC", int64_t(1024));
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    prof.pop();

    wait_for_drain();

    auto        lock       = prof.consumer()->lock_shared();
    auto const &thread_map = prof.consumer()->thread_data();

    auto const *node = find_node_any_thread(thread_map, "annot_int_test");
    REQUIRE(node != nullptr);
    REQUIRE(node->annotations.count("MC") > 0);
    REQUIRE(node->annotations.at("MC") == "256");
    REQUIRE(node->annotations.at("NC") == "1024");
}

TEST_CASE("Annotations appear in print output", "[profiler][annotations]") {
    auto &prof = Profiler::instance();

    prof.push("annot_prn_test");
    annotate("alg", "DOT");
    annotate("N", int64_t(100));
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    prof.pop();

    wait_for_drain();

    std::ostringstream oss;
    prof.print(false, oss);

    std::string output = oss.str();
    REQUIRE(output.find("annot_prn_test") != std::string::npos);
    REQUIRE(output.find("alg=DOT") != std::string::npos);
}

TEST_CASE("ProfileAnnotate macro works", "[profiler][annotations]") {
    {
        LabeledSection("macro_annot_test");
        ProfileAnnotate("key", "value");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    wait_for_drain();

    auto        lock       = Profiler::instance().consumer()->lock_shared();
    auto const &thread_map = Profiler::instance().consumer()->thread_data();

    auto const *node = find_node_any_thread(thread_map, "macro_annot_test");
    REQUIRE(node != nullptr);
    REQUIRE(node->annotations.count("key") > 0);
    REQUIRE(node->annotations.at("key") == "value");
}

// ── Per-site caches for formatted zone names and annotation values ──────────
//
// A formatted zone name used to be formatted and interned (under the string table's lock) on every
// entry, and so did every annotation's key and string value. Each call site now caches its ids per
// thread, keyed by the arguments' values. These cases pin that the cache never hands back a stale
// name or value.

namespace {

struct Recording {
    explicit Recording(bool on) : _was(Profiler::instance().enabled()) { Profiler::instance().set_enabled(on); }
    Recording(Recording const &)            = delete;
    Recording &operator=(Recording const &) = delete;
    ~Recording() { Profiler::instance().set_enabled(_was); }

  private:
    bool _was;
};

void cached_zone(std::vector<std::string> const &c, int n) {
    LabeledSection("cached_zone: {} n={}", fmt::join(c, ","), n);
}

/// A zone argument with no value to key on: only fmt can say what it prints as.
struct Opaque {
    int value;
};

/// Calls of every zone named @p name, on every thread: each thread has a tree of its own.
uint64_t zone_calls_below(AggNode const &node, std::string const &name) {
    uint64_t total = node.name == name ? node.call_count : 0;
    for (auto const &child : node.children) {
        total += zone_calls_below(*child.second, name);
    }
    return total;
}

uint64_t zone_calls(std::string const &name) {
    auto     lock  = Profiler::instance().consumer()->lock_shared();
    uint64_t total = 0;
    for (auto const &thread : Profiler::instance().consumer()->thread_data()) {
        total += zone_calls_below(thread.second.root, name);
    }
    return total;
}

} // namespace

template <>
struct fmt::formatter<Opaque> : fmt::formatter<int> {
    auto format(Opaque const &o, fmt::format_context &ctx) const -> fmt::format_context::iterator {
        return fmt::format_to(ctx.out(), "<{}>", o.value);
    }
};

TEST_CASE("A formatted zone name is cached per call site and stays right", "[profiler][annotations][cache]") {
    Recording const                on(true);
    std::vector<std::string> const ij{"i", "j"}, kl{"k", "l"};

    cached_zone(ij, 1);
    cached_zone(kl, 2);
    cached_zone(ij, 1);
    cached_zone(kl, 3);
    std::thread([&] {
        cached_zone(ij, 1);
        cached_zone(kl, 3);
    }).join();
    Profiler::instance().flush();

    CHECK(zone_calls("cached_zone: i,j n=1") == 3);
    CHECK(zone_calls("cached_zone: k,l n=2") == 1);
    CHECK(zone_calls("cached_zone: k,l n=3") == 2);
}

TEST_CASE("A zone argument with no value to key on is formatted every time", "[profiler][annotations][cache]") {
    Recording const on(true);
    for (int const v : {1, 2, 1}) {
        LabeledSection("unkeyed zone {}", Opaque{v});
    }
    // A single-pass range cannot be read once for the key and again to format.
    std::istringstream words("a b");
    {
        LabeledSection("single-pass zone {}",
                       fmt::join(std::istream_iterator<std::string>(words), std::istream_iterator<std::string>(), "+"));
    }
    Profiler::instance().flush();
    CHECK(zone_calls("unkeyed zone <1>") == 2);
    CHECK(zone_calls("unkeyed zone <2>") == 1);
    CHECK(zone_calls("single-pass zone a+b") == 1);
}

TEST_CASE("A conditional between two same-length literals annotates each value", "[profiler][annotations][cache]") {
    // ``i % 2 ? "T" : "N"`` has the type of a single literal, char const[2]. A cache that trusted the
    // type would record whichever value it met first at every later entry.
    Recording const on(true);
    for (int i = 0; i < 4; ++i) {
        LabeledSection("ternary annotation zone {}", i);
        ProfileAnnotate("trans", i % 2 != 0 ? "T" : "N");
    }
    Profiler::instance().flush();

    auto        lock = Profiler::instance().consumer()->lock_shared();
    auto const &map  = Profiler::instance().consumer()->thread_data();
    for (int i = 0; i < 4; ++i) {
        auto const *node = find_node_any_thread(map, fmt::format("ternary annotation zone {}", i));
        REQUIRE(node != nullptr);
        CHECK(node->annotations.at("trans") == (i % 2 != 0 ? "T" : "N"));
    }
}

TEST_CASE("Annotation values and zone arguments are not evaluated while recording is off", "[profiler][annotations][cache]") {
    Recording const off(false);
    int             evaluated = 0;
    auto const      value     = [&] {
        ++evaluated;
        return int64_t{1};
    };
    {
        LabeledSection("lazy zone {}", value());
        ProfileAnnotate("lazy", value());
        ProfileAnnotate("lazy string", std::to_string(value()));
    }
    CHECK(evaluated == 0);
}
