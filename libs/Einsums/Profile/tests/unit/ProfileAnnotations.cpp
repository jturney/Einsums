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

#endif
