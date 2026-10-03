//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Einsums' zones and its callers' land in one tree. Waggle's own tests live in libs/Waggle/tests;
// these need a zone opened inside libEinsums.

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Profile/Profile.hpp>

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

namespace {

/// The zone named @p name anywhere below @p node.
std::optional<waggle::SnapshotNode> find_zone(waggle::SnapshotNode const &node, std::string const &name) {
    std::vector<waggle::SnapshotNode> pending{node};
    while (!pending.empty()) {
        auto const current = pending.back();
        pending.pop_back();
        for (auto const &child : current.children()) {
            if (child.name() == name) {
                return child;
            }
            pending.push_back(child);
        }
    }
    return std::nullopt;
}

} // namespace

TEST_CASE("A zone opened in the library nests under the caller's zone", "[profiler]") {
    // The per-thread channel was reached through an inline function holding a thread_local, and
    // -fvisibility-inlines-hidden gives every shared object its own copy: this executable and the
    // library each had a channel for the thread, with its own ring and depth, and the library's
    // zones landed at the root instead of under the zone that called them. dgemv opens its zone in
    // the library, not in anything inlined here.
    {
        WAGGLE_ZONE("caller zone across libraries");
        std::array<double, 4> const a{1.0, 2.0, 3.0, 4.0};
        std::array<double, 2> const x{1.0, 1.0};
        std::array<double, 2>       y{};
        einsums::blas::vendor::dgemv('n', 2, 2, 1.0, a.data(), 2, x.data(), 1, 0.0, y.data(), 1);
    }

    auto const                          snapshot = waggle::Snapshot::take();
    std::optional<waggle::SnapshotNode> caller;
    for (auto const &thread : snapshot.threads()) {
        caller = find_zone(thread.root, "caller zone across libraries");
        if (caller) {
            break;
        }
    }
    REQUIRE(caller);
    bool nested = false;
    for (auto const &child : caller->children()) {
        nested = nested || child.name() == "dgemv";
    }
    CHECK(nested);
}
