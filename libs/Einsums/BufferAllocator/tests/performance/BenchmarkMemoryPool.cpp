//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The decisive gate for DESIGN-memory-pool.md: what a multi-megabyte allocation
// costs in DLPNO-(T0)'s ACTUAL regime.
//
// That regime is monotonic fill - hundreds of multi-megabyte blocks allocated
// and ALL HELD LIVE, so every one lands on fresh OS pages. A hot alloc/free
// loop is explicitly not the question: macOS libmalloc recycles a just-freed
// 7 MiB block in about 100 ns, which says nothing about the 78-150 us per
// tensor measured inside T0.
//
// Three configurations, same sizes and counts:
//
//   system    the platform's own aligned allocator, the allocation path
//             einsums used before mimalloc became required
//   mimalloc  einsums::memory::aligned_alloc, the current Layer 0, which is
//             mimalloc's default heap
//   pool      MemoryPool, an exclusive mimalloc arena reserved up front
//
// Each is reported alloc-only and again with a first-touch write per page,
// because the two costs move independently: an allocator can be fast and still
// hand back memory whose first write faults.

#include <Einsums/BufferAllocator/MemoryPool.hpp>

#if defined(EINSUMS_WINDOWS)
#    include <malloc.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;

namespace {

constexpr size_t kMiB = 1024ULL * 1024ULL;

/// How the platform's own aligned allocator is spelled in the report.
#if defined(EINSUMS_WINDOWS)
constexpr char const *kSystemLabel = "system (_aligned_malloc)";
#else
constexpr char const *kSystemLabel = "system (posix_memalign)";
#endif

/// Page size used for the first-touch sweep. Deliberately the smallest common
/// page (4 KiB) so the sweep touches every page on every platform.
constexpr size_t kTouchStride = 4096;

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void touch(void *ptr, size_t bytes) {
    auto *p = static_cast<unsigned char volatile *>(ptr);
    for (size_t off = 0; off < bytes; off += kTouchStride) {
        p[off] = 1;
    }
}

struct Timing {
    double alloc_ms{0};
    double touch_ms{0};
    double free_ms{0};
};

/// One monotonic fill: @p count blocks of @p bytes, every one held live until
/// the fill is done.
Timing fill(size_t count, size_t bytes, std::function<void *(size_t)> const &alloc, std::function<void(void *)> const &release,
            bool do_touch) {
    std::vector<void *> held;
    held.reserve(count);

    Timing t;

    auto const alloc_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < count; i++) {
        void *p = alloc(bytes);
        if (p == nullptr) {
            std::printf("  allocation %zu of %zu FAILED\n", i, count);
            break;
        }
        held.push_back(p);
    }
    t.alloc_ms = elapsed_ms(alloc_start);

    if (do_touch) {
        auto const touch_start = std::chrono::steady_clock::now();
        for (void *p : held) {
            touch(p, bytes);
        }
        t.touch_ms = elapsed_ms(touch_start);
    }

    auto const free_start = std::chrono::steady_clock::now();
    for (void *p : held) {
        release(p);
    }
    t.free_ms = elapsed_ms(free_start);

    return t;
}

void report(char const *label, size_t count, size_t bytes, Timing const &t) {
    double const per_alloc_us = t.alloc_ms * 1000.0 / static_cast<double>(count);
    std::printf("  %-24s alloc %8.2f ms (%7.2f us/block)  touch %8.2f ms  free %8.2f ms\n", label, t.alloc_ms, per_alloc_us, t.touch_ms,
                t.free_ms);
    std::fflush(stdout);
    (void)bytes;
}

// The platform's own aligned allocator, which is what einsums called before
// mimalloc became required. Windows has no posix_memalign, and its aligned
// blocks must go back through _aligned_free rather than free().
void *system_alloc(size_t bytes) {
#if defined(EINSUMS_WINDOWS)
    return _aligned_malloc(bytes, 64);
#else
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) {
        return nullptr;
    }
    return p;
#endif
}

void system_free(void *p) {
#if defined(EINSUMS_WINDOWS)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

void run_regime(size_t count, size_t bytes, bool do_touch) {
    std::printf("\n%zu blocks x %zu MiB (%zu MiB live)%s\n", count, bytes / kMiB, count * bytes / kMiB,
                do_touch ? ", with first touch" : "");

    report(kSystemLabel, count, bytes, fill(count, bytes, system_alloc, system_free, do_touch));

    report("mimalloc (default heap)", count, bytes,
           fill(
               count, bytes, [](size_t n) { return memory::aligned_alloc(n); }, [](void *p) { memory::aligned_free(p); }, do_touch));

    {
        // Reserved for the whole fill up front, which is how a caller with a
        // chunk plan would size it.
        MemoryPool pool(count * bytes, "bench");
        auto       carve   = [&pool](size_t n) { return pool.allocate(n); };
        auto       release = [&pool](void *p) { pool.deallocate(p); };

        report("pool (cold arena)", count, bytes, fill(count, bytes, carve, release, do_touch));
        // Second pass over the same arena: the pages are committed and warm,
        // which is the state every chunk after the first sees.
        report("pool (warm arena)", count, bytes, fill(count, bytes, carve, release, do_touch));
        if (pool.arenas() != 1) {
            std::printf("  NOTE: pool grew to %zu arenas\n", pool.arenas());
        }
    }
}

} // namespace

EINSUMS_TEST_CASE("MemoryPool monotonic fill", "[performance][memory][pool]") {
    // 7 MiB and 26 MiB are the two block sizes DLPNO-(T0)'s chunk state is
    // built from. Counts are chosen to hold about 1.7 GiB live in each regime,
    // enough for the allocator's behavior at scale without crowding the box.
    run_regime(256, 7 * kMiB, false);
    run_regime(256, 7 * kMiB, true);
    run_regime(64, 26 * kMiB, false);
    run_regime(64, 26 * kMiB, true);
}
