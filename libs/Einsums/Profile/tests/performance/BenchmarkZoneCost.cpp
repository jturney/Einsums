//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkZoneCost.cpp
/// @brief What one profiling zone costs, and which part of the entry path the cost is in.
///
/// A zone is timed as a whole, recording on and off, and then the pieces it is built from are timed
/// one at a time: the clock, the singleton, the shared atomics that count the profiler's own
/// overhead, the counter backend, the thread's ring buffer, the event copy and the string table.
/// The whole-zone numbers say how far there is to go; the pieces say where it goes.
///
/// Every batch stays well under the ring buffer's capacity and the consumer drains it between
/// batches, outside the timed region. A batch that fills the ring measures the consumer's
/// throughput instead of the producer's entry cost, and every case converges on the same number.

#include <Einsums/Profile/Consumer.hpp>
#include <Einsums/Profile/CounterBackend.hpp>
#include <Einsums/Profile/Profile.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#    include <mach/mach_time.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#    include <x86intrin.h>
#endif

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace prof = einsums::profile;

namespace {

/// Operations per timed batch: 4096 zones are 8192 events, an eighth of the 65536-slot ring.
constexpr int kOps  = 4096;
constexpr int kReps = 50;

/// Keeps the compiler from moving work across a batch boundary or deleting a loop body.
inline void barrier() {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

std::uint64_t volatile g_sink = 0;

template <typename T>
void keep(T const &v) {
    g_sink = g_sink + static_cast<std::uint64_t>(v);
}

struct Result {
    double min_ns;
    double median_ns;
};

/// Nanoseconds per call of @p op, over batches of kOps, with the consumer drained between batches.
template <typename Op>
Result per_op(Op &&op) {
    auto               &profiler = prof::Profiler::instance();
    std::vector<double> ns;
    for (int rep = 0; rep <= kReps; ++rep) {
        profiler.flush();
        barrier();
        auto const t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kOps; ++i) {
            op(i);
            barrier();
        }
        auto const t1 = std::chrono::steady_clock::now();
        barrier();
        if (rep > 0) { // the first batch is a warmup
            ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / kOps);
        }
    }
    std::ranges::sort(ns);
    return {.min_ns = ns.front(), .median_ns = ns[ns.size() / 2]};
}

void show(std::string_view label, Result r) {
    fmt::println("[ZoneCost {:46s}] min {:7.1f} ns   median {:7.1f} ns", label, r.min_ns, r.median_ns);
}

/// Restores the profiler's recording flag on scope exit.
struct Recording {
    explicit Recording(bool on) : _was(prof::Profiler::instance().enabled()) { prof::Profiler::instance().set_enabled(on); }
    Recording(Recording const &)            = delete;
    Recording &operator=(Recording const &) = delete;
    ~Recording() { prof::Profiler::instance().set_enabled(_was); }

  private:
    bool _was;
};

std::atomic<std::uint64_t> g_shared_counter{0};

/// The access pattern the per-thread ring buffer used to be reached through.
auto thread_slot() -> std::shared_ptr<int> & {
    thread_local auto slot = std::make_shared<int>(1);
    return slot;
}

} // namespace

TEST_CASE("Bench ZoneCost: one zone, whole and in pieces", "[Profile][ZoneCost][benchmark]") {
    fmt::println("[ZoneCost] sizeof(Event) = {} bytes, ring capacity = {} events", sizeof(prof::Event), prof::kRingBufferCapacity);

    show("empty loop", per_op([](int i) { keep(i); }));

    // ── The whole zone ──────────────────────────────────────────────────
    {
        Recording const off(false);
        show("zone, literal name, recording OFF", per_op([](int) { LabeledSection("bench zone"); }));
    }
    {
        Recording const on(true);
        show("zone, literal name, recording on", per_op([](int) { LabeledSection("bench zone"); }));

        // The shape cg::einsum opens on every call: three joined index lists, formatted and
        // interned per entry.
        std::vector<std::string> const c{"i", "j"}, a{"i", "k"}, b{"k", "j"};
        show("zone, formatted like cg::einsum",
             per_op([&](int) { LabeledSection("cg::einsum: {} <- {} ; {}", fmt::join(c, ","), fmt::join(a, ","), fmt::join(b, ",")); }));

        // Annotations attach to the open zone, so open one around the batch.
        LabeledSection("bench annotate host");
        show("annotate, int64", per_op([](int i) { ProfileAnnotate("bench key", static_cast<std::int64_t>(i)); }));
        show("annotate, string literal", per_op([](int) { ProfileAnnotate("bench key", "bench value"); }));
    }

    // ── The pieces a recorded zone is built from ────────────────────────
    // What a recorded zone is built from, each done twice (push and pop) unless noted. The ones
    // marked "was" are what the entry path did before it read the CPU counter and stopped timing
    // itself; they stay here as the reference the new path is measured against.
    show("steady_clock::now()  [was x4 per zone]", per_op([](int) { keep(std::chrono::steady_clock::now().time_since_epoch().count()); }));
    // What a Tracy-style design would read instead of steady_clock: the raw hardware counter,
    // converted to nanoseconds by the consumer rather than on every event.
#if defined(__aarch64__) || defined(_M_ARM64)
#    if !defined(_MSC_VER)
    std::uint64_t freq = 0;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    fmt::println("[ZoneCost] cntvct_el0 runs at {:.1f} MHz, one tick = {:.1f} ns", freq / 1e6, 1e9 / static_cast<double>(freq));
    show("raw counter read (cntvct_el0)  [x2]", per_op([](int) {
             std::uint64_t v = 0;
             asm volatile("mrs %0, cntvct_el0" : "=r"(v));
             keep(v);
         }));
#    endif
#elif defined(__x86_64__) || defined(_M_X64)
    show("raw counter read (rdtsc)  [x2]", per_op([](int) { keep(__rdtsc()); }));
#endif
#if defined(__APPLE__)
    show("mach_absolute_time()", per_op([](int) { keep(mach_absolute_time()); }));
#endif
    show("Profiler::instance()  [x2]", per_op([](int) { keep(reinterpret_cast<std::uintptr_t>(&prof::Profiler::instance())); }));
    show("fetch_add on a shared atomic  [was x4]", per_op([](int) { g_shared_counter.fetch_add(1, std::memory_order_relaxed); }));
    show("counter backend read  [x2 if a backend is active]", per_op([](int) {
             std::array<std::uint64_t, prof::kNumCounterSlots> values{};
             prof::get_counter_backend().read(values);
             keep(values[0]);
         }));
    // The pattern the per-thread ring used to be reached through: a function-local thread_local
    // with dynamic initialization, a shared_ptr returned by reference. It is now a plain pointer.
    show("thread_local lookup, dynamic init  [was x2+]", per_op([](int) { keep(reinterpret_cast<std::uintptr_t>(thread_slot().get())); }));
    {
        // A private ring, so the profiler's own is not touched. Push and pop together keep it
        // from filling; the pair is two event copies, which is what a zone's push and pop cost.
        static prof::EventRingBuffer ring;
        prof::Event                  evt{};
        show("event copy in and out of a ring  [x1 pair]", per_op([&](int i) {
                 evt.line = i;
                 (void)ring.try_push(evt);
                 prof::Event out{};
                 (void)ring.try_pop(out);
                 keep(out.line);
             }));
    }
    show("string intern of a known literal  [0 on a literal zone]",
         per_op([](int) { keep(prof::Profiler::instance().string_table().intern("bench zone")); }));
}

/// Nanoseconds per call of @p op on the slowest of @p threads threads running it at once, each
/// over a batch of kOps, the consumer drained between batches.
template <typename Op>
Result team_per_op(int threads, Op const &op) {
    auto               &profiler = prof::Profiler::instance();
    std::vector<double> ns;
    for (int rep = 0; rep <= 20; ++rep) {
        std::atomic<int>         ready{0};
        std::atomic<bool>        go{false};
        std::vector<double>      per_thread(threads, 0.0);
        std::vector<std::thread> team;
        team.reserve(threads);
        for (int t = 0; t < threads; ++t) {
            team.emplace_back([&, t] {
                {
                    LabeledSection("bench warm"); // registers the thread's ring outside the timing
                }
                ready.fetch_add(1);
                while (!go.load(std::memory_order_acquire)) {
                }
                auto const t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < kOps; ++i) {
                    op(i);
                    barrier();
                }
                auto const t1 = std::chrono::steady_clock::now();
                per_thread[t] = std::chrono::duration<double, std::nano>(t1 - t0).count() / kOps;
            });
        }
        while (ready.load() < threads) {
        }
        go.store(true, std::memory_order_release);
        for (auto &th : team) {
            th.join();
        }
        profiler.flush();
        if (rep > 0) {
            ns.push_back(*std::ranges::max_element(per_thread));
        }
    }
    std::ranges::sort(ns);
    return {.min_ns = ns.front(), .median_ns = ns[ns.size() / 2]};
}

TEST_CASE("Bench ZoneCost: zones on several threads at once", "[Profile][ZoneCost][benchmark]") {
    // Every recorded zone does four relaxed fetch_adds on counters shared by the whole process. On
    // one thread that is cheap; across a team the cache lines bounce between cores. The pieces are
    // run the same way so the scaling can be pinned on the one that has it. This M4 has 4
    // performance cores, so 8 threads also lands some on efficiency cores.
    Recording const on(true);
    for (int const threads : {1, 2, 4, 8}) {
        auto const label = [&](std::string_view what) { return fmt::format("{}, {} thread(s)", what, threads); };
        show(label("zone, literal"), team_per_op(threads, [](int) { LabeledSection("bench zone"); }));
        show(label("4 x fetch_add, shared atomic"), team_per_op(threads, [](int) {
                 for (int k = 0; k < 4; ++k) {
                     g_shared_counter.fetch_add(1, std::memory_order_relaxed);
                 }
             }));
        show(label("4 x increment, thread_local"), team_per_op(threads, [](int) {
                 thread_local std::uint64_t local = 0;
                 for (int k = 0; k < 4; ++k) {
                     local = local + 1;
                     barrier();
                 }
                 keep(local);
             }));
        show(label("4 x steady_clock::now()"), team_per_op(threads, [](int) {
                 for (int k = 0; k < 4; ++k) {
                     keep(std::chrono::steady_clock::now().time_since_epoch().count());
                 }
             }));
    }
}
