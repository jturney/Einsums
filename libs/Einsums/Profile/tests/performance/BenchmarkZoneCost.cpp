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

#include <Einsums/Performance.hpp>
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
#include <waggle/Consumer.hpp>
#include <waggle/CounterBackend.hpp>

#if defined(__APPLE__)
#    include <mach/mach_time.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#    include <x86intrin.h>
#endif

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <sys/resource.h>
#endif

#include <Einsums/Testing.hpp>

/// An index list printed as fmt::join prints it, through a type the name cache cannot key on.
struct Unkeyed {
    std::vector<std::string> const &v;
};

template <>
struct fmt::formatter<Unkeyed> : fmt::formatter<std::string_view> {
    auto format(Unkeyed const &u, fmt::format_context &ctx) const -> fmt::format_context::iterator {
        return fmt::format_to(ctx.out(), "{}", fmt::join(u.v, ","));
    }
};

using namespace einsums;
namespace prof = waggle;

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

/// Into the benchmark database as `zone-cost <label>`, the median as the value: the figure this
/// file reads, since the minimum of a few-nanosecond operation is mostly a lucky batch.
void publish(std::string_view label, char const *metric, Result r, int reps) {
    std::string_view  trimmed = label.substr(std::min(label.find_first_not_of(' '), label.size()));
    std::string const name    = fmt::format("zone-cost {}", trimmed);
    double const      median  = r.median_ns / 1000.0;
    performance::publish_benchmark_result(name.c_str(), metric,
                                          performance::TimingStats{median, r.min_ns / 1000.0, median, 0.0, 0.0, reps});
}

void show(std::string_view label, Result r) {
    fmt::println("[ZoneCost {:46s}] min {:7.1f} ns   median {:7.1f} ns", label, r.min_ns, r.median_ns);
    publish(label, "t_op", r, kReps);
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
        show("zone, literal name, recording OFF", per_op([](int) { WAGGLE_ZONE("bench zone"); }));
    }
    {
        Recording const on(true);
        show("zone, literal name, recording on", per_op([](int) { WAGGLE_ZONE("bench zone"); }));

        // The shape cg::einsum opens on every call: three joined index lists, formatted and
        // interned per entry.
        std::vector<std::string> const c{"i", "j"}, a{"i", "k"}, b{"k", "j"};
        show("zone, formatted, fmt::join (cached per site)",
             per_op([&](int) { WAGGLE_ZONE("cg::einsum: {} <- {} ; {}", fmt::join(c, ","), fmt::join(a, ","), fmt::join(b, ",")); }));
        // The same name through an argument with no value to key on, which is formatted every entry:
        // what every formatted zone cost before the cache.
        show("zone, formatted, no key (formats every entry)",
             per_op([&](int) { WAGGLE_ZONE("cg::einsum: {} <- {} ; {}", Unkeyed{c}, Unkeyed{a}, Unkeyed{b}); }));
        // The cached path's pieces, for the three joined index lists above.
        {
            std::array<char, prof::site_cache::kNameKeyCapacity> buffer{};
            std::size_t                                          size = 0;
            show("  key: build from three joined lists", per_op([&](int) {
                     prof::site_cache::KeyWriter w{.pos = buffer.data(), .end = buffer.data() + buffer.size()};
                     prof::site_cache::write_key(w, fmt::join(c, ","));
                     prof::site_cache::write_key(w, fmt::join(a, ","));
                     prof::site_cache::write_key(w, fmt::join(b, ","));
                     size = static_cast<std::size_t>(w.pos - buffer.data());
                     keep(size);
                 }));
            std::string const key(buffer.data(), size);
            fmt::println("[ZoneCost] that key is {} bytes", key.size());
            show("  key: hash", per_op([&](int) { keep(prof::site_cache::StringKeyHash{}(key)); }));
            prof::site_cache::IdCache cache;
            cache.emplace(key, 7);
            show("  key: find in the cache (hit)", per_op([&](int) { keep(cache.find(std::string_view(key))->second); }));
        }
        show("zone, formatted from an int (cached per site)", per_op([](int i) { WAGGLE_ZONE("gemv<TransA={}>", (i & 1) != 0); }));

        // Annotations attach to the open zone, so open one around the batch.
        WAGGLE_ZONE("bench annotate host");
        show("annotate, int64", per_op([](int i) { WAGGLE_ANNOTATE("bench key", static_cast<std::int64_t>(i)); }));
        show("annotate, string literal", per_op([](int) { WAGGLE_ANNOTATE("bench key", "bench value"); }));
        show("annotate, string from a conditional",
             per_op([](int i) { WAGGLE_ANNOTATE("bench key", (i & 1) != 0 ? "left" : "right side"); }));
        // The unmacroed call interns key and value on every call, under the string table's lock.
        show("annotate(), string, interned per call", per_op([](int) { prof::annotate("bench key", "bench value"); }));
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
                    WAGGLE_ZONE("bench warm"); // registers the thread's ring outside the timing
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
        show(label("empty loop"), team_per_op(threads, [](int i) { keep(i); }));
        {
            Recording const off(false);
            show(label("zone, literal, recording OFF"), team_per_op(threads, [](int) { WAGGLE_ZONE("bench zone"); }));
        }
        show(label("zone, literal"), team_per_op(threads, [](int) { WAGGLE_ZONE("bench zone"); }));
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

TEST_CASE("Bench ZoneCost: the consumer", "[Profile][ZoneCost][benchmark]") {
    // The consumer aggregates every event into the report's tree on a thread of its own. What it
    // spends per event bounds the rate a program can record at before the rings fill and events
    // are dropped; what it spends while nothing records is paid by every program.
    Recording const on(true);
    auto           &profiler = prof::Profiler::instance();

    // flush() drains on the calling thread, so timing it times the aggregation. A batch of 16384
    // zones is half a ring and takes a fraction of the consumer thread's 1 ms nap to produce, so the
    // consumer thread rarely takes part of it first; the median over batches is reported.
    constexpr int kZones = 16384;
    auto const    drain  = [&](std::string_view label, int events_per_zone, auto &&produce) {
        std::vector<double> ns;
        for (int rep = 0; rep <= 30; ++rep) {
            profiler.flush();
            for (int i = 0; i < kZones; ++i) {
                produce(i);
            }
            auto const t0 = std::chrono::steady_clock::now();
            profiler.flush();
            auto const t1 = std::chrono::steady_clock::now();
            if (rep > 0) {
                ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / (kZones * events_per_zone));
            }
        }
        std::ranges::sort(ns);
        fmt::println("[ZoneCost consumer: {:38s}] min {:7.1f} ns   median {:7.1f} ns   per event", label, ns.front(), ns[ns.size() / 2]);
        publish(fmt::format("consumer, {}", label), "t_per_event", Result{.min_ns = ns.front(), .median_ns = ns[ns.size() / 2]}, 30);
    };
    drain("literal zones", 2, [](int) { WAGGLE_ZONE("bench consumer zone"); });
    drain("zones with an int annotation", 3, [](int i) {
        WAGGLE_ZONE("bench consumer zone");
        WAGGLE_ANNOTATE("bench key", static_cast<std::int64_t>(i));
    });
    drain("zones with a string annotation", 3, [](int) {
        WAGGLE_ZONE("bench consumer zone");
        WAGGLE_ANNOTATE("bench key", "bench value");
    });

    // CPU the whole process uses while nothing is recorded: the consumer's 1 ms naps and anything
    // else the profiler runs in the background.
    auto const cpu_seconds = [] {
#if defined(_WIN32)
        // FILETIME counts 100 ns intervals.
        FILETIME   creation{}, exit{}, kernel{}, user{};
        auto const ticks = [](FILETIME const &ft) {
            return static_cast<double>((static_cast<std::uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime);
        };
        GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user);
        return 1e-7 * (ticks(kernel) + ticks(user));
#else
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        return static_cast<double>(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) +
               1e-6 * static_cast<double>(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec);
#endif
    };
    profiler.flush();
    double const c0 = cpu_seconds();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    double const c1 = cpu_seconds();
    fmt::println("[ZoneCost consumer: idle process CPU over 2 s              ] {:.2f} ms of CPU per second ({:.3f}% of one core)",
                 1000.0 * (c1 - c0) / 2.0, 100.0 * (c1 - c0) / 2.0);
    // Microseconds of CPU per wall second, so the database's microsecond unit still reads true.
    double const idle_us = 1e6 * (c1 - c0) / 2.0;
    performance::publish_benchmark_result("zone-cost consumer, idle process CPU", "idle_cpu_us_per_s",
                                          performance::TimingStats{idle_us, idle_us, idle_us, 0.0, 0.0, 1});
}

namespace {

/// Nanoseconds per push into a private ring that a second thread drains as the pushes arrive, the
/// way the profiler's producer and consumer share one. @p work_ns of busy work per popped event
/// stands in for the consumer's aggregation: zero keeps the ring near empty, enough of it keeps the
/// ring full so every push takes the full-ring path.
Result ring_with_consumer(int work_ns, std::uint64_t &dropped) {
    constexpr int       kPushes = 1 << 22; // 64 rings' worth
    auto                ring    = std::make_unique<prof::EventRingBuffer>();
    std::vector<double> ns;
    dropped = 0;
    for (int rep = 0; rep <= 5; ++rep) {
        std::atomic<bool> done{false};
        std::thread       consumer([&] {
            auto const spin = [&] {
                auto const until = std::chrono::steady_clock::now() + std::chrono::nanoseconds(work_ns);
                while (work_ns > 0 && std::chrono::steady_clock::now() < until) {
                }
            };
            while (!done.load(std::memory_order_acquire) || !ring->empty()) {
                ring->drain([&](prof::Event const &evt) {
                    keep(evt.line);
                    spin();
                });
            }
        });
        std::uint64_t     lost = 0;
        prof::Event       evt{};
        auto const        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kPushes; ++i) {
            evt.line = i;
            if (!ring->try_push(evt)) {
                ++lost;
            }
            keep(ring->past_half());
            barrier();
        }
        auto const t1 = std::chrono::steady_clock::now();
        done.store(true, std::memory_order_release);
        consumer.join();
        if (rep > 0) {
            ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / kPushes);
            dropped += lost;
        }
    }
    std::ranges::sort(ns);
    return {.min_ns = ns.front(), .median_ns = ns[ns.size() / 2]};
}

} // namespace

TEST_CASE("Bench ZoneCost: a ring with its consumer running", "[Profile][ZoneCost][benchmark]") {
    // The other cases drain between batches so they time the producer alone. Here the consumer runs
    // at the same time, as it does in a program: the producer and consumer write the two ends of
    // one ring, and what crosses between their cores is what this measures.
    for (int const work : {0, 50}) {
        std::uint64_t dropped = 0;
        auto const    r       = ring_with_consumer(work, dropped);
        show(fmt::format("ring push + past_half, consumer {} ns/event", work), r);
        fmt::println("[ZoneCost]   {:.1f}% of those pushes found the ring full", 100.0 * static_cast<double>(dropped) / (5.0 * (1 << 22)));
    }
}

TEST_CASE("Bench ZoneCost: zones while the rings overflow", "[Profile][ZoneCost][benchmark]") {
    // Each thread opens and closes zones faster than one consumer can aggregate them, for eight
    // rings' worth of events without a drain in between, so the rings stay full and most events are
    // dropped. A profiler is under the most pressure to stay cheap here, and every thread is
    // pushing at once.
    Recording const on(true);
    auto           &profiler = prof::Profiler::instance();
    constexpr int   kZones   = 4 * static_cast<int>(prof::kRingBufferCapacity);
    for (int const threads : {1, 2, 4, 8}) {
        std::vector<double> ns;
        std::uint64_t       dropped = 0;
        for (int rep = 0; rep <= 5; ++rep) {
            profiler.flush();
            auto const               before = profiler.consumer()->dropped_count();
            std::atomic<int>         ready{0};
            std::atomic<bool>        go{false};
            std::vector<double>      per_thread(threads, 0.0);
            std::vector<std::thread> team;
            for (int t = 0; t < threads; ++t) {
                team.emplace_back([&, t] {
                    {
                        WAGGLE_ZONE("bench warm");
                    }
                    ready.fetch_add(1);
                    while (!go.load(std::memory_order_acquire)) {
                    }
                    auto const t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < kZones; ++i) {
                        WAGGLE_ZONE("bench overflow zone");
                        barrier();
                    }
                    auto const t1 = std::chrono::steady_clock::now();
                    per_thread[t] = std::chrono::duration<double, std::nano>(t1 - t0).count() / kZones;
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
                dropped += profiler.consumer()->dropped_count() - before;
            }
        }
        std::ranges::sort(ns);
        show(fmt::format("zone, rings overflowing, {} thread(s)", threads), Result{.min_ns = ns.front(), .median_ns = ns[ns.size() / 2]});
        fmt::println("[ZoneCost]   {:.1f}% of events dropped", 100.0 * static_cast<double>(dropped) / (5.0 * 2.0 * kZones * threads));
    }
}
