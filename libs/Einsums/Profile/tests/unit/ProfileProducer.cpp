//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The producer half of the profiler: the clock events are stamped with, the per-thread counts,
// and the overhead figure. A zone used to read steady_clock four times and bump four counters every
// thread shared; it now reads the CPU's counter twice and writes only its own thread's memory.
// These cases pin what that must not have cost.

#include <Einsums/Config.hpp>

#include <Einsums/Profile/Profile.hpp>
#include <Waggle/Clock.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "Profiler.hpp"

using namespace waggle;

namespace {

/// Restores the profiler's recording flag on scope exit.
struct Recording {
    explicit Recording(bool on) : _was(Profiler::instance().enabled()) { Profiler::instance().set_enabled(on); }
    Recording(Recording const &)            = delete;
    Recording &operator=(Recording const &) = delete;
    ~Recording() { Profiler::instance().set_enabled(_was); }

  private:
    bool _was;
};

} // namespace

TEST_CASE("TickClock is monotonic", "[profiler][clock]") {
    std::uint64_t last = TickClock::now();
    for (int i = 0; i < 100000; ++i) {
        std::uint64_t const t = TickClock::now();
        REQUIRE(t >= last);
        last = t;
    }
}

TEST_CASE("TickClock converts to steady_clock time", "[profiler][clock]") {
    auto const &clock = TickClock::instance();
    INFO("source " << clock.source << ", " << clock.ns_per_tick << " ns per tick");
    REQUIRE(clock.ns_per_tick > 0.0);

    SECTION("an instant maps to the steady_clock instant it was read at") {
        auto const          before = std::chrono::steady_clock::now();
        std::uint64_t const ticks  = TickClock::now();
        auto const          after  = std::chrono::steady_clock::now();
        auto const          at     = clock.to_time_point(ticks);
        // Within the bracket, give or take the anchor's own skew and the counter's resolution.
        auto const slack = std::chrono::microseconds(50);
        CHECK(at >= before - slack);
        CHECK(at <= after + slack);
    }

    SECTION("an interval measures what steady_clock measures") {
        auto const          t0 = std::chrono::steady_clock::now();
        std::uint64_t const c0 = TickClock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::uint64_t const c1 = TickClock::now();
        auto const          t1 = std::chrono::steady_clock::now();

        double const steady_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double const ticks_ns  = std::chrono::duration<double, std::nano>(clock.to_time_point(c1) - clock.to_time_point(c0)).count();
        // The tick interval sits inside the steady_clock one; on x86 the rate is measured, so
        // allow it half a percent.
        CHECK(ticks_ns <= steady_ns + 1000.0);
        CHECK(ticks_ns >= 0.995 * steady_ns - 10000.0);
    }
}

TEST_CASE("Zone counts are exact across threads, exited ones included", "[profiler][counts]") {
    // The counts used to be fetch_adds on counters every thread shared. They are now each thread's
    // own, summed when read, and a thread's count must survive the thread.
    Recording const on(true);
    auto           &prof    = Profiler::instance();
    auto const      pushes0 = prof.total_push_count();
    auto const      pops0   = prof.total_pop_count();

    constexpr int            kThreads = 6;
    constexpr int            kZones   = 1000;
    std::vector<std::thread> team;
    team.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        team.emplace_back([] {
            for (int i = 0; i < kZones; ++i) {
                WAGGLE_ZONE("counted zone");
            }
        });
    }
    for (auto &th : team) {
        th.join();
    }
    {
        WAGGLE_ZONE("counted zone on the caller");
    }

    CHECK(prof.total_push_count() == pushes0 + std::uint64_t{kThreads} * kZones + 1);
    CHECK(prof.total_pop_count() == pops0 + std::uint64_t{kThreads} * kZones + 1);
}

TEST_CASE("A pop with nothing open is not counted", "[profiler][counts]") {
    Recording const on(true);
    auto           &prof  = Profiler::instance();
    auto const      pops0 = prof.total_pop_count();
    // Run on a fresh thread, whose depth is certainly zero.
    std::thread([&] { prof.pop(); }).join();
    CHECK(prof.total_pop_count() == pops0);
}

TEST_CASE("The overhead figure is calibrated, not accumulated", "[profiler][overhead]") {
    auto &prof = Profiler::instance();
    // Measured once and then fixed, so two reads agree exactly.
    double const push = prof.avg_push_overhead_ns();
    double const pop  = prof.avg_pop_overhead_ns();
    CHECK(push == prof.avg_push_overhead_ns());
    CHECK(pop == prof.avg_pop_overhead_ns());
    // Positive and far below the steady_clock-based path it replaced (about 36 ns a push).
    CHECK(push > 0.0);
    CHECK(pop > 0.0);
    CHECK(push < 1000.0);
    CHECK(pop < 1000.0);
}
