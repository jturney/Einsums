//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// A zone's domain is found by name lookup from where the zone is written.

#include <Waggle/Config.hpp>

#include <Waggle/Waggle.hpp>

#include <catch2/catch_test_macros.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// Two libraries, as they would declare themselves. What follows stands in for their headers, so it
// keeps external linkage.
// NOLINTBEGIN(misc-use-internal-linkage)
namespace lib_a {
WAGGLE_DEFINE_DOMAIN("lib_a")

constexpr std::string_view current() {
    return WAGGLE_CURRENT_DOMAIN;
}

namespace detail {
constexpr std::string_view nested() {
    return WAGGLE_CURRENT_DOMAIN;
}
} // namespace detail

/// A zone in a template: lookup happens where it is defined, not where it is used.
template <typename T>
void zone_in_template() {
    WAGGLE_ZONE("domains: lib_a template");
}

inline void zone_in_inline() {
    WAGGLE_ZONE("domains: lib_a inline");
}
} // namespace lib_a

namespace lib_b {
WAGGLE_DEFINE_DOMAIN("lib_b")

constexpr std::string_view current() {
    return WAGGLE_CURRENT_DOMAIN;
}

inline void zone() {
    WAGGLE_ZONE("domains: lib_b");
}
} // namespace lib_b

// An application namespace that brings both libraries into view: its zones are its own.
namespace app {
using namespace lib_a;
using namespace lib_b;

constexpr std::string_view current() {
    return WAGGLE_CURRENT_DOMAIN;
}
} // namespace app

// Application code at global scope, with both libraries in view: the unnamed domain, and no
// ambiguity between them.
using namespace lib_a;
using namespace lib_b;

constexpr std::string_view global_current() {
    return WAGGLE_CURRENT_DOMAIN;
}

// NOLINTEND(misc-use-internal-linkage)

static_assert(lib_a::current() == "lib_a");
static_assert(lib_a::detail::nested() == "lib_a");
static_assert(lib_b::current() == "lib_b");
static_assert(app::current().empty());
static_assert(global_current().empty());

namespace {

/// The zone named @p name anywhere in a fresh snapshot.
std::optional<std::string> domain_of(std::string const &name) {
    auto const                        snapshot = waggle::Snapshot::take();
    std::vector<waggle::SnapshotNode> pending;
    for (auto const &thread : snapshot.threads()) {
        pending.push_back(thread.root);
    }
    while (!pending.empty()) {
        auto const node = pending.back();
        pending.pop_back();
        for (auto const &child : node.children()) {
            if (child.name() == name) {
                return std::string(child.domain());
            }
            pending.push_back(child);
        }
    }
    return std::nullopt;
}

} // namespace

TEST_CASE("A zone belongs to the library whose namespace it is written in", "[profiler][domains]") {
    waggle::set_enabled(true);
    std::thread([] {
        lib_a::zone_in_template<int>();
        lib_a::zone_in_inline();
        lib_b::zone();
        WAGGLE_ZONE("domains: application");
    }).join();

    CHECK(domain_of("domains: lib_a template") == "lib_a");
    CHECK(domain_of("domains: lib_a inline") == "lib_a");
    CHECK(domain_of("domains: lib_b") == "lib_b");
    CHECK(domain_of("domains: application") == "");

    // Merging threads keeps each zone's domain.
    auto const merged = waggle::Snapshot::take(true);
    auto const zone   = merged.find(0, "domains: lib_b");
    REQUIRE(zone);
    CHECK(zone->domain() == "lib_b");
}
