//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The rules that decide whether a hardware profile applies, checked without a file. Which file
// is found, and what a process does with it, is exercised end to end by the Python tests, each in
// a fresh process.

#include <Einsums/Hardware/Machine.hpp>
#include <Einsums/Hardware/Profile.hpp>

#include <string>
#include <string_view>
#include <utility>

#include <Einsums/Testing.hpp>

using namespace einsums;

namespace {

hardware::MachineIdentity a_machine() {
    hardware::MachineIdentity machine;
    machine.cpu            = "Test CPU";
    machine.physical_cores = 8;
    machine.logical_cores  = 16;
    machine.caches         = hardware::CacheSizes{.l1 = 65536, .l2 = 1048576, .l3 = 33554432};
    machine.memory_gib     = 64;
    machine.simd           = "x86-64-v3";
    return machine;
}

hardware::SoftwareIdentity a_build() {
    return hardware::SoftwareIdentity{.major = 2, .minor = 1, .patch = 3, .blas = "OpenBLAS 0.3.28 (libopenblas.so)"};
}

/// Accepts one option, as the parser's check accepts the tunable ones.
std::string accept_tunables(std::string_view name) {
    return name == "einsums:graph:tiling-budget-share" ? std::string{} : std::string{"not a tunable"};
}

json::Value document_for(hardware::MachineIdentity const &machine, hardware::SoftwareIdentity const &software) {
    json::Object root;
    root.set("einsums_profile", hardware::profile_format);
    root.set("machine", hardware::machine_section(machine));
    root.set("software", hardware::software_section(software));
    return json::Value{std::move(root)};
}

hardware::ProfileState state_of(json::Value const &document, bool named = false) {
    return hardware::evaluate_profile(document, "test.json", a_machine(), a_build(), named, accept_tunables).state;
}

} // namespace

TEST_CASE("Hardware profile - every identity field moves the fingerprint", "[Hardware][Profile]") {
    auto const base = hardware::fingerprint(a_machine());
    CHECK(base.size() == 16);
    CHECK(hardware::fingerprint(a_machine()) == base);

    auto const differs = [&](auto &&edit) {
        auto machine = a_machine();
        edit(machine);
        return hardware::fingerprint(machine) != base;
    };
    CHECK(differs([](auto &m) { m.cpu = "Other CPU"; }));
    CHECK(differs([](auto &m) { m.physical_cores += 1; }));
    CHECK(differs([](auto &m) { m.logical_cores += 1; }));
    CHECK(differs([](auto &m) { m.caches.l1 *= 2; }));
    CHECK(differs([](auto &m) { m.caches.l2 *= 2; }));
    CHECK(differs([](auto &m) { m.caches.l3 *= 2; }));
    CHECK(differs([](auto &m) { m.memory_gib += 1; }));
    CHECK(differs([](auto &m) { m.simd = "x86-64-v4"; }));
}

TEST_CASE("Hardware profile - a machine section reads back as the identity it was written from", "[Hardware][Profile]") {
    auto const  section = hardware::machine_section(a_machine());
    std::string error;
    auto const  read = hardware::read_machine_section(section, error);
    REQUIRE(read.has_value());
    CHECK(*read == a_machine());
}

TEST_CASE("Hardware profile - which documents apply", "[Hardware][Profile]") {
    using hardware::ProfileState;
    CHECK(state_of(document_for(a_machine(), a_build())) == ProfileState::Loaded);

    auto patch = a_build();
    patch.patch += 4;
    CHECK(state_of(document_for(a_machine(), patch)) == ProfileState::Loaded);

    auto minor = a_build();
    minor.minor += 1;
    CHECK(state_of(document_for(a_machine(), minor)) == ProfileState::Stale);

    auto major = a_build();
    major.major += 1;
    CHECK(state_of(document_for(a_machine(), major)) == ProfileState::Stale);

    auto blas = a_build();
    blas.blas = "Accelerate";
    CHECK(state_of(document_for(a_machine(), blas)) == ProfileState::Stale);

    auto other = a_machine();
    other.logical_cores *= 2;
    CHECK(state_of(document_for(other, a_build())) == ProfileState::Foreign);
    CHECK(state_of(document_for(other, a_build()), /*named=*/true) == ProfileState::LoadedForeign);

    // Another build makes a named file stale too: naming a file overrides the machine, not the
    // software.
    CHECK(state_of(document_for(other, minor), /*named=*/true) == ProfileState::Stale);
}

TEST_CASE("Hardware profile - what makes a document invalid", "[Hardware][Profile]") {
    using hardware::ProfileState;
    auto const with = [](std::string const &key, json::Value value) {
        auto const   base = document_for(a_machine(), a_build());
        json::Object root;
        for (auto const &existing : base.as_object().keys()) {
            root.set(existing, existing == key ? value : *base.as_object().peek(existing));
        }
        if (!root.contains(key)) {
            root.set(key, std::move(value));
        }
        return json::Value{std::move(root)};
    };

    CHECK(state_of(json::Value{json::Array{}}) == ProfileState::Invalid);
    CHECK(state_of(with("einsums_profile", json::Value{std::int64_t{2}})) == ProfileState::Invalid);
    CHECK(state_of(with("surprise", json::Value{true})) == ProfileState::Invalid);
    CHECK(state_of(with("software", json::Value{std::string{"2.1.3"}})) == ProfileState::Invalid);

    auto         edited = hardware::machine_section(a_machine());
    json::Object tampered;
    for (auto const &key : edited.keys()) {
        tampered.set(key, key == "memory_gib" ? json::Value{std::int64_t{128}} : *edited.peek(key));
    }
    auto const verdict = hardware::evaluate_profile(with("machine", json::Value{std::move(tampered)}), "test.json", a_machine(), a_build(),
                                                    false, accept_tunables);
    CHECK(verdict.state == ProfileState::Invalid);
    CHECK_THAT(verdict.reason, Catch::Matchers::ContainsSubstring("fingerprint"));

    auto extra = hardware::machine_section(a_machine());
    extra.set("hostname", "node17");
    auto const unknown = hardware::evaluate_profile(with("machine", json::Value{std::move(extra)}), "test.json", a_machine(), a_build(),
                                                    false, accept_tunables);
    CHECK(unknown.state == ProfileState::Invalid);
    CHECK_THAT(unknown.reason, Catch::Matchers::ContainsSubstring("machine.hostname"));
}

TEST_CASE("Hardware profile - the options section", "[Hardware][Profile]") {
    using hardware::ProfileState;
    auto const with_options = [](json::Object options) {
        auto         document = document_for(a_machine(), a_build());
        json::Object root;
        for (auto const &key : document.as_object().keys()) {
            root.set(key, *document.as_object().peek(key));
        }
        root.set("options", std::move(options));
        return json::Value{std::move(root)};
    };
    auto const verdict = [](json::Value const &document, hardware::ProfileOptionCheck check) {
        return hardware::evaluate_profile(document, "test.json", a_machine(), a_build(), false, check);
    };

    json::Object tunable;
    tunable.set("einsums:graph:tiling-budget-share", 0.25);
    CHECK(verdict(with_options(tunable), accept_tunables).state == ProfileState::Loaded);

    // Without a registered check, as below the parser, no option may come from a profile.
    CHECK(verdict(with_options(tunable), nullptr).state == ProfileState::Invalid);

    json::Object refused;
    refused.set("einsums:log:level", std::int64_t{1});
    auto const not_tunable = verdict(with_options(refused), accept_tunables);
    CHECK(not_tunable.state == ProfileState::Invalid);
    CHECK_THAT(not_tunable.reason, Catch::Matchers::ContainsSubstring("einsums:log:level"));

    json::Object nested;
    nested.set("einsums:graph:tiling-budget-share", json::Object{});
    CHECK(verdict(with_options(nested), accept_tunables).state == ProfileState::Invalid);
}
