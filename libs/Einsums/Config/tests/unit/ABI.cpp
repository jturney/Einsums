//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/ABI.hpp>

#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEST_CASE("world is one record, and the same one every time", "[abi]") {
    sealed::WorldInfo const &a = sealed::world();
    sealed::WorldInfo const &b = sealed::world();

    // Same object, not merely equal: callers compare `identity` by value, and a
    // fresh record per call would make that comparison meaningless.
    REQUIRE(&a == &b);
    REQUIRE(a.identity != nullptr);
    REQUIRE(a.struct_size == sizeof(sealed::WorldInfo));
}

// BuildModeProbe.cpp, compiled twice with opposite NDEBUG settings.
extern "C" std::uint64_t einsums_probe_ndebug();
extern "C" std::uint64_t einsums_probe_debug();

TEST_CASE("the config fingerprint separates debug from release", "[abi]") {
    // The case that made this necessary: on MSVC a debug stage module and a
    // release library are linked against different C runtimes, with different
    // heaps and different standard-library layouts, so mixing them corrupts
    // rather than misbehaves. Before this term existed the two compared equal
    // and the handshake waved them through.
    //
    // Two compilations of ONE source, differing only in NDEBUG. Anything else
    // asserted about a single fingerprint value would restate the header rather
    // than test it.
    REQUIRE(einsums_probe_ndebug() != einsums_probe_debug());

    // ...and the library's own value is one of the two, not a third thing,
    // which is what says the probe measures the same function the world does.
    std::uint64_t const mine = sealed::config_fingerprint();
    REQUIRE((mine == einsums_probe_ndebug() || mine == einsums_probe_debug()));
}

TEST_CASE("the caller's fingerprints match the library's", "[abi]") {
    // This is the whole point of the fingerprints. This test compiles against
    // the same headers the library was built from, so the two must agree; a
    // stage module compiled against STALE headers is exactly the case where
    // they would not, and that is what the handshake refuses.
    REQUIRE(sealed::world().config_fingerprint == sealed::config_fingerprint());
    REQUIRE(sealed::world().layout_fingerprint != 0);
}

TEST_CASE("world reports a usable identity string set", "[abi]") {
    sealed::WorldInfo const &w = sealed::world();

    REQUIRE(w.version_major == EINSUMS_VERSION_MAJOR);
    REQUIRE(w.version_minor == EINSUMS_VERSION_MINOR);
    REQUIRE(w.version_patch == EINSUMS_VERSION_PATCH);

    for (char const *s : {w.version_string, w.git_commit, w.compiler_id, w.library_path}) {
        REQUIRE(s != nullptr);
    }
    REQUIRE(std::string(w.version_string).find(std::to_string(EINSUMS_VERSION_MAJOR)) != std::string::npos);
    REQUIRE(std::string(w.compiler_id) != "Unknown");
}

TEST_CASE("the library path names a library, not the test binary", "[abi]") {
    // dladdr resolves from a symbol INSIDE libEinsums, so a shared build must
    // report the library. If this ever names the test executable, the identity
    // has been inlined into the caller and every world comparison is worthless.
    std::string const path = sealed::world().library_path;
    if (path.empty()) {
        SKIP("platform did not report a library path");
    }
    REQUIRE(path.find("Einsums") != std::string::npos);
}

TEST_CASE("stage-module registration is a set, keyed by name", "[abi]") {
    REQUIRE_FALSE(sealed::stage_module_registered("abi_test_stage"));

    REQUIRE(sealed::register_stage_module("abi_test_stage"));
    REQUIRE(sealed::stage_module_registered("abi_test_stage"));

    // Registering twice reports "already there" rather than duplicating: the
    // loader may legitimately re-import a module.
    REQUIRE_FALSE(sealed::register_stage_module("abi_test_stage"));

    std::vector<std::string> const names = sealed::registered_stage_modules();
    REQUIRE(std::count(names.begin(), names.end(), "abi_test_stage") == 1);

    // An unrelated name is unaffected, and a null name is a no-op rather than a
    // crash: this is called from module init code that may be generated.
    REQUIRE_FALSE(sealed::stage_module_registered("never_registered"));
    REQUIRE_FALSE(sealed::register_stage_module(nullptr));
    REQUIRE_FALSE(sealed::stage_module_registered(nullptr));
}

TEST_CASE("this process maps exactly one libEinsums", "[abi]") {
    std::vector<std::string> const libs = sealed::mapped_einsums_libraries();

    if (libs.empty()) {
        SKIP("platform cannot enumerate loaded images");
    }

    // The diagnostic's whole value is that >1 means two worlds. A plain test
    // binary linking one shared library must therefore report exactly one, or
    // the signal is noise.
    INFO("mapped: " << [&] {
        std::string joined;
        for (auto const &l : libs) {
            joined += "\n  " + l;
        }
        return joined;
    }());
    REQUIRE(libs.size() == 1);
    REQUIRE(libs.front().find("Einsums") != std::string::npos);
}
