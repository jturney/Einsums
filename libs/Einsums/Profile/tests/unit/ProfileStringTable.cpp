//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Profile/StringTable.hpp>

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <string>
#include <thread>

using namespace einsums::profile;

TEST_CASE("StringTable interns each string once", "[profiler][stringtable]") {
    StringTable st;

    uint32_t const a = st.intern("alpha");
    uint32_t const b = st.intern("beta");

    REQUIRE(a != b);
    REQUIRE(st.intern("alpha") == a);
    REQUIRE(st.size() == 2);

    REQUIRE(st.get(a) == "alpha");
    REQUIRE(st.get(b) == "beta");
}

TEST_CASE("StringTable ids start at zero and increase by one", "[profiler][stringtable]") {
    StringTable st;

    REQUIRE(st.intern("first") == 0);
    REQUIRE(st.intern("second") == 1);
    REQUIRE(st.intern("third") == 2);
}

TEST_CASE("StringTable interns the empty string like any other", "[profiler][stringtable]") {
    StringTable st;

    uint32_t const id = st.intern("");
    REQUIRE(st.get(id).empty());
    REQUIRE(st.size() == 1);
}

// The regression this file exists for.
//
// Consumer resolves every id it sees straight out of an event popped from a
// ring buffer, so it cannot establish that the id is one this table issued.
// get() used to index the deque unconditionally, which is undefined for an
// out-of-range id: it returned a reference to arbitrary bytes that the caller
// then used as a std::string. Consumer::process_annotate hashes that value into
// a map key, so the wild pointer was dereferenced immediately and killed the
// process from the consumer thread - the Linux coverage leg died exactly there,
// in _Hash_bytes under process_annotate.
//
// Static libEinsums builds are how ids crossed tables in the first place: they
// fold a private profiler into every extension module. That duplication is a
// separate problem. This asserts only that the table survives being asked, and
// says so, because a telemetry consumer must never be able to abort the program
// over an annotation it cannot name.
TEST_CASE("StringTable answers an id it never issued", "[profiler][stringtable]") {
    StringTable st;

    SECTION("empty table") {
        REQUIRE(st.get(0) == StringTable::unknown_string());
        REQUIRE(st.get(12345) == StringTable::unknown_string());
    }

    SECTION("one past the end") {
        uint32_t const id = st.intern("only");
        REQUIRE(st.get(id) == "only");
        REQUIRE(st.get(id + 1) == StringTable::unknown_string());
    }

    SECTION("the largest representable id") {
        st.intern("only");
        REQUIRE(st.get(UINT32_MAX) == StringTable::unknown_string());
    }

    SECTION("the placeholder is distinguishable from an interned empty string") {
        uint32_t const id = st.intern("");
        REQUIRE(st.get(id) != StringTable::unknown_string());
    }
}

TEST_CASE("StringTable reads stay valid while another thread interns", "[profiler][stringtable]") {
    // References handed out by get() point into a deque that keeps growing.
    // deque never invalidates references to existing elements on push_back,
    // which is the property the Consumer relies on when it holds a
    // `std::string const &` past the shared lock.
    StringTable st;

    uint32_t const     first = st.intern("stable");
    std::string const &held  = st.get(first);

    std::atomic<bool> go{false};
    std::thread       writer([&] {
        while (!go.load(std::memory_order_acquire)) {
        }
        for (int i = 0; i < 2000; ++i) {
            st.intern("filler-" + std::to_string(i));
        }
    });

    go.store(true, std::memory_order_release);

    for (int i = 0; i < 2000; ++i) {
        REQUIRE(held == "stable");
        REQUIRE(st.get(first) == "stable");
    }

    writer.join();

    REQUIRE(held == "stable");
    REQUIRE(st.size() >= 2001);
}
