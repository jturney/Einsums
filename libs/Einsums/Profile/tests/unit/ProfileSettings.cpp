//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Profile/Profile.hpp>
#include <Waggle/Diagnostics.hpp>
#include <Waggle/Settings.hpp>

#include <catch2/catch_test_macros.hpp>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace waggle;

namespace {

/// An environment holding exactly @p vars.
EnvironmentReader environment(std::map<std::string, std::string> vars) {
    return [vars = std::move(vars)](char const *name) -> std::optional<std::string> {
        auto it = vars.find(name);
        return it == vars.end() ? std::nullopt : std::optional<std::string>(it->second);
    };
}

AggNode const *find_named(AggNode const &node, std::string const &name) {
    if (node.name == name) {
        return &node;
    }
    for (auto const &child : node.children) {
        if (auto const *found = find_named(*child.second, name)) {
            return found;
        }
    }
    return nullptr;
}

} // namespace

TEST_CASE("Settings default to what Einsums' options default to", "[profiler][settings]") {
    SettingsStore const store;
    auto const         &s = store.current();
    CHECK(s.record);
    CHECK(s.report);
    CHECK(s.report_file == "profile.txt");
    CHECK_FALSE(s.report_append);
    CHECK_FALSE(s.server);
    CHECK(s.port == 19216);
    CHECK(s.max_distinct_children == 256);
}

TEST_CASE("The environment sets what no library set", "[profiler][settings]") {
    SettingsStore store;
    auto const    problems = store.apply_environment(environment(
        {{"WAGGLE_DISABLE", "1"}, {"WAGGLE_REPORT_FILE", "env.txt"}, {"WAGGLE_PORT", "20000"}, {"WAGGLE_REPORT_APPEND", "on"}}));
    CHECK(problems.empty());
    CHECK_FALSE(store.current().record); // WAGGLE_DISABLE is the inverse of record
    CHECK(store.current().report_file == "env.txt");
    CHECK(store.current().port == 20000);
    CHECK(store.current().report_append);
}

TEST_CASE("An explicit setting beats the environment, whichever comes first", "[profiler][settings]") {
    SettingsUpdate update;
    update.report_file = "explicit.txt";

    SettingsStore env_first;
    (void)env_first.apply_environment(environment({{"WAGGLE_REPORT_FILE", "env.txt"}}));
    CHECK(env_first.configure(update).empty());
    CHECK(env_first.current().report_file == "explicit.txt");

    SettingsStore explicit_first;
    CHECK(explicit_first.configure(update).empty());
    (void)explicit_first.apply_environment(environment({{"WAGGLE_REPORT_FILE", "env.txt"}}));
    CHECK(explicit_first.current().report_file == "explicit.txt");
}

TEST_CASE("The first library to set a setting keeps it", "[profiler][settings]") {
    SettingsStore  store;
    SettingsUpdate first;
    first.port = 20001;
    CHECK(store.configure(first).empty());

    SettingsUpdate same;
    same.port = 20001;
    CHECK(store.configure(same).empty()); // agreeing is not a conflict

    SettingsUpdate second;
    second.port        = 20002;
    auto const refused = store.configure(second);
    REQUIRE(refused.size() == 1);
    CHECK(refused.front().find("port") != std::string::npos);
    CHECK(store.current().port == 20001);

    // Settings a library left alone are still free for another to set.
    SettingsUpdate other;
    other.report_detailed = true;
    CHECK(store.configure(other).empty());
    CHECK(store.current().report_detailed);
}

TEST_CASE("A value that does not parse is reported and skipped", "[profiler][settings]") {
    SettingsStore store;
    auto const    problems = store.apply_environment(environment({{"WAGGLE_SERVER", "maybe"}, {"WAGGLE_PORT", "12ab"}}));
    CHECK(problems.size() == 2);
    CHECK_FALSE(store.current().server);
    CHECK(store.current().port == 19216);
}

TEST_CASE("Overriding replaces a setting whoever set it", "[profiler][settings]") {
    SettingsStore  store;
    SettingsUpdate first;
    first.save = "a.json";
    CHECK(store.configure(first).empty());

    SettingsUpdate restore;
    restore.save = "";
    store.override_settings(restore);
    CHECK(store.current().save.empty());
}

// Libraries share one profiler, so one finishing must not stop it for the others: only the last
// finalize writes the outputs and stops the consumer. Einsums holds one reference for this whole
// test run, so a client released here leaves the profiler recording.
TEST_CASE("A library's finalize leaves the profiler running for the others", "[profiler][settings]") {
    auto &prof = Profiler::instance();
    prof.init({.name = "test-client"});
    prof.finalize("test-client");

    REQUIRE(prof.enabled());
    std::string const name = "settings: recording after a client's finalize";
    {
        ScopedZone const zone(name);
    }
    prof.flush();

    auto           lock = prof.consumer()->lock_shared();
    AggNode const *node = nullptr;
    for (auto const &thread : prof.consumer()->thread_data()) {
        node = find_named(thread.second.root, name);
        if (node != nullptr) {
            break;
        }
    }
    REQUIRE(node != nullptr);
    CHECK(node->call_count == 1);
}

// A refused setting must be visible, and through the host's logger when it installed one, not lost
// to a stderr nobody reads.
TEST_CASE("A refused setting is reported through the diagnostic handler", "[profiler][settings]") {
    std::vector<std::pair<DiagnosticLevel, std::string>> seen;
    set_diagnostic_handler([&seen](DiagnosticLevel level, std::string_view message) { seen.emplace_back(level, std::string(message)); });

    auto          &prof = Profiler::instance();
    SettingsUpdate first;
    first.save = "first-library.json";
    prof.override_settings(first);
    SettingsUpdate second;
    second.save = "second-library.json";
    prof.configure(second);

    set_diagnostic_handler({});
    SettingsUpdate restore;
    restore.save = "";
    prof.override_settings(restore);

    REQUIRE(seen.size() == 1);
    CHECK(seen.front().first == DiagnosticLevel::Warning);
    CHECK(seen.front().second.find("first-library.json") != std::string::npos);
    CHECK(seen.front().second.find("second-library.json") != std::string::npos);
}
