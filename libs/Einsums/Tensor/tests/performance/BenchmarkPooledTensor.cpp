//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// What a tensor construction costs in DLPNO-(T0)'s regime, end to end.
//
// The allocator microbenchmark next door (BufferAllocator's
// BenchmarkMemoryPool) shows the raw malloc call is 1-3 us for a multi-megabyte
// block on every configuration, which is nowhere near the 78-150 us per tensor
// measured inside T0. This benchmark shows where the rest goes: an owned
// RuntimeTensor is backed by std::vector, so construction VALUE-INITIALIZES
// every element, touching every page of fresh memory. A pooled tensor carves
// from an arena whose pages are already committed and does not initialize at
// all unless asked.
//
// Same regime as the allocator benchmark: monotonic fill, every tensor held
// live, multi-megabyte shapes.

#include <Einsums/Tensor/PooledTensor.hpp>

#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;

namespace {

constexpr size_t kMiB = 1024ULL * 1024ULL;

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void report(char const *label, size_t count, double ms) {
    std::printf("  %-28s %8.2f ms (%7.2f us/tensor)\n", label, ms, ms * 1000.0 / static_cast<double>(count));
    std::fflush(stdout);
}

void run_regime(size_t count, size_t mib) {
    using Held = std::vector<std::unique_ptr<RuntimeTensor<double>>>;

    std::vector<size_t> const dims{mib * kMiB / sizeof(double)};
    std::printf("\n%zu tensors x %zu MiB (%zu MiB live)\n", count, mib, count * mib);

    // Held through unique_ptr, not by value: these tensor types have no move
    // constructor, so a vector of values would deep-copy on every push and
    // measure the copy instead of the construction.
    {
        Held held;
        held.reserve(count);
        auto const start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < count; i++) {
            held.push_back(std::make_unique<RuntimeTensor<double>>("owned", dims));
        }
        report("RuntimeTensor (owned)", count, elapsed_ms(start));
    }

    {
        MemoryPool pool(count * mib * kMiB, "tensors");

        auto carve = [&pool, &dims](bool zero) {
            auto t = std::make_unique<RuntimeTensor<double>>(RuntimeTensor<double>::deferred_alloc, "pooled", dims);
            pool.place(*t);
            if (zero) {
                t->zero();
            }
            return t;
        };

        Held held;
        held.reserve(count);
        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < count; i++) {
            held.push_back(carve(false));
        }
        report("pooled empty (cold arena)", count, elapsed_ms(start));
        held.clear();

        held.reserve(count);
        start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < count; i++) {
            held.push_back(carve(false));
        }
        report("pooled empty (warm arena)", count, elapsed_ms(start));
        held.clear();

        held.reserve(count);
        start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < count; i++) {
            held.push_back(carve(true));
        }
        report("pooled zeros (warm arena)", count, elapsed_ms(start));

        if (pool.arenas() != 1) {
            std::printf("  NOTE: pool grew to %zu arenas\n", pool.arenas());
        }
    }
}

} // namespace

EINSUMS_TEST_CASE("Pooled tensor construction", "[performance][tensor][pool]") {
    run_regime(256, 7);
    run_regime(64, 26);
}
