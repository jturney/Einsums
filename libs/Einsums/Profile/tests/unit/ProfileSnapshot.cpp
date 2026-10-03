//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Snapshots, reset and the recording switch, through Waggle's public interface only.

#include <Einsums/Config.hpp>

#include <Einsums/Profile/Profile.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

#include <Einsums/Testing.hpp>

namespace {

/// Restores the recording switch on scope exit.
struct Recording {
    explicit Recording(bool on) : _was(waggle::enabled()) { waggle::set_enabled(on); }
    Recording(Recording const &)            = delete;
    Recording &operator=(Recording const &) = delete;
    ~Recording() { waggle::set_enabled(_was); }

  private:
    bool _was;
};

/// Run @p body on a thread of its own, so its zones sit directly below that thread's root.
template <typename Body>
void on_fresh_thread(Body &&body) {
    std::thread(std::forward<Body>(body)).join();
}

} // namespace

TEST_CASE("A snapshot holds zones with their statistics and annotations", "[profiler][snapshot]") {
    Recording const on(true);
    on_fresh_thread([] {
        for (int i = 0; i < 3; ++i) {
            waggle::ScopedZone const outer("snapshot: outer");
            waggle::annotate("phase", "setup");
            waggle::annotate("flops", std::int64_t{100});
            waggle::ScopedZone const inner("snapshot: inner");
        }
    });

    auto const snapshot = waggle::Snapshot::take();
    auto const outer    = snapshot.find("snapshot: outer");
    auto const inner    = snapshot.find("snapshot: outer/snapshot: inner");
    REQUIRE(outer);
    REQUIRE(inner);
    CHECK_FALSE(snapshot.find("snapshot: outer/no such zone"));
    CHECK_FALSE(snapshot.find("snapshot: inner")); // not at the root

    auto const o = outer->stats();
    auto const n = inner->stats();
    CHECK(o.call_count == 3);
    CHECK(n.call_count == 3);
    CHECK(o.inclusive_ns == o.exclusive_ns + n.inclusive_ns);
    CHECK(o.exclusive_min_ns <= o.exclusive_max_ns);
    std::uint64_t calls = 0;
    for (auto const bucket : o.histogram) {
        calls += bucket;
    }
    CHECK(calls == 3);

    // Text holds every annotation's latest value, numeric ones too.
    auto const text = outer->annotations();
    REQUIRE(text.size() == 2);
    CHECK(text[0] == std::pair<std::string, std::string>("phase", "setup"));
    CHECK(text[1] == std::pair<std::string, std::string>("flops", "100"));
    auto const numeric = outer->numeric_annotations();
    REQUIRE(numeric.size() == 1);
    CHECK(numeric[0].key == "flops");
    CHECK(numeric[0].count == 3);
    CHECK(numeric[0].total == 300.0);

    REQUIRE(outer->children().size() == 1);
    CHECK(outer->children()[0].name() == "snapshot: inner");
}

TEST_CASE("A merged snapshot combines the same zone from every thread", "[profiler][snapshot]") {
    Recording const on(true);
    for (int const calls : {2, 5}) {
        on_fresh_thread([calls] {
            for (int i = 0; i < calls; ++i) {
                waggle::ScopedZone const outer("merge: outer");
                waggle::annotate("count", std::int64_t{1});
                waggle::ScopedZone const inner("merge: inner");
            }
        });
    }

    auto const merged = waggle::Snapshot::take(true);
    REQUIRE(merged.threads().size() == 1);
    CHECK(merged.threads()[0].id == 0);
    auto const outer = merged.find(0, "merge: outer");
    auto const inner = merged.find(0, "merge: outer/merge: inner");
    REQUIRE(outer);
    REQUIRE(inner);
    CHECK(outer->stats().call_count == 7);
    CHECK(inner->stats().call_count == 7);
    REQUIRE(outer->numeric_annotations().size() == 1);
    CHECK(outer->numeric_annotations()[0].count == 7);

    // Per thread, each keeps its own count.
    auto const    separate = waggle::Snapshot::take();
    std::uint64_t total    = 0;
    int           found    = 0;
    for (size_t t = 0; t < separate.threads().size(); ++t) {
        if (auto node = separate.find(t, "merge: outer")) {
            total += node->stats().call_count;
            ++found;
        }
    }
    CHECK(found == 2);
    CHECK(total == 7);
}

TEST_CASE("A reset keeps open zones, timed from the reset, and drops the rest", "[profiler][snapshot]") {
    Recording const                       on(true);
    uint32_t const                        id_before = waggle::intern_string("reset: open");
    std::atomic<bool>                     opened{false}, reset_done{false};
    std::chrono::steady_clock::time_point reset_started;

    std::thread worker([&] {
        {
            waggle::ScopedZone const closed("reset: closed");
        }
        waggle::ScopedZone const open("reset: open");
        // Long enough that time from the push would show plainly.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        opened = true;
        while (!reset_done) {
            std::this_thread::yield();
        }
        {
            waggle::ScopedZone const after("reset: after");
        }
    });
    while (!opened) {
        std::this_thread::yield();
    }
    reset_started = std::chrono::steady_clock::now();
    waggle::reset();
    reset_done = true;
    worker.join();
    auto const elapsed = std::chrono::steady_clock::now() - reset_started;

    auto const snapshot = waggle::Snapshot::take();
    CHECK_FALSE(snapshot.find("reset: closed"));
    auto const open = snapshot.find("reset: open");
    REQUIRE(open);
    auto const stats = open->stats();
    CHECK(stats.call_count == 1);
    // Timed from the reset: no more than the time since it began, well under the 50 ms before.
    CHECK(stats.exclusive_ns <= std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed + std::chrono::milliseconds(1)).count());
    auto const after = snapshot.find("reset: open/reset: after");
    REQUIRE(after);
    CHECK(after->stats().call_count == 1);

    // Ids are identities, not statistics: a reset keeps them.
    CHECK(waggle::intern_string("reset: open") == id_before);
}

// ScopedZone once let the switch decide at exit whether to close: a zone entered with recording
// off and left with it on closed the zone around it, and one entered on and left off stayed open.
TEST_CASE("A zone closes exactly when it opened, whatever the switch does in between", "[profiler][snapshot]") {
    Recording const on(true);
    on_fresh_thread([] {
        waggle::ScopedZone const outer("pairing: outer");
        {
            waggle::set_enabled(false);
            waggle::ScopedZone const skipped("pairing: skipped");
            waggle::set_enabled(true);
        } // must not close outer
        waggle::ScopedZone const inner("pairing: inner");
    });
    on_fresh_thread([] {
        {
            waggle::ScopedZone const left_off("pairing: left off");
            waggle::set_enabled(false);
        } // must close, though recording is off now
        waggle::set_enabled(true);
        waggle::ScopedZone const next("pairing: next");
    });

    auto const snapshot = waggle::Snapshot::take();
    CHECK(snapshot.find("pairing: outer/pairing: inner"));
    CHECK_FALSE(snapshot.find("pairing: inner"));
    CHECK_FALSE(snapshot.find("pairing: outer/pairing: skipped"));
    REQUIRE(snapshot.find("pairing: left off"));
    CHECK(snapshot.find("pairing: left off")->stats().call_count == 1);
    CHECK(snapshot.find("pairing: next")); // a sibling, not a child
    CHECK_FALSE(snapshot.find("pairing: left off/pairing: next"));
}

TEST_CASE("The inline recording switch is the collector's", "[profiler][snapshot]") {
    Recording const     on(true);
    std::int32_t const *flag = waggle_enabled_flag();
    REQUIRE(flag != nullptr);
    CHECK(flag == waggle_enabled_flag());
    waggle::set_enabled(false);
    CHECK(std::atomic_ref<std::int32_t>(*const_cast<std::int32_t *>(flag)).load() == 0);
    CHECK_FALSE(waggle::enabled());
    waggle::set_enabled(true);
    CHECK(std::atomic_ref<std::int32_t>(*const_cast<std::int32_t *>(flag)).load() == 1);
    CHECK(waggle::enabled());
}

TEST_CASE("A caller built against fewer statistics gets only those", "[profiler][snapshot]") {
    Recording const on(true);
    on_fresh_thread([] { waggle::ScopedZone const zone("stats size: zone"); });

    waggle_snapshot   *snapshot = waggle_snapshot_take(0);
    waggle_node const *zone     = nullptr;
    std::string const  path     = "stats size: zone";
    for (size_t t = 0; zone == nullptr && t < waggle_snapshot_thread_count(snapshot); ++t) {
        zone = waggle_snapshot_find(snapshot, t, path.data(), path.size());
    }
    REQUIRE(zone != nullptr);

    // As if the caller's struct ended after call_count: nothing past it may be written.
    waggle_node_stats stats;
    std::memset(&stats, 0xAB, sizeof(stats));
    stats.size = static_cast<uint32_t>(offsetof(waggle_node_stats, exclusive_ns));
    waggle_node_stats_get(zone, &stats);
    CHECK(stats.size == offsetof(waggle_node_stats, exclusive_ns));
    CHECK(stats.call_count == 1);
    waggle_node_stats untouched;
    std::memset(&untouched, 0xAB, sizeof(untouched));
    CHECK(std::memcmp(&stats.exclusive_ns, &untouched.exclusive_ns, sizeof(stats) - offsetof(waggle_node_stats, exclusive_ns)) == 0);

    waggle_snapshot_release(snapshot);
}
