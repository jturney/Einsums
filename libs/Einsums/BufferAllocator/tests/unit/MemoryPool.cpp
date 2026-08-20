//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/MemoryPool.hpp>
#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/Options/Get.hpp>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using namespace einsums;

namespace {

constexpr size_t kMiB = 1024ULL * 1024ULL;

bool aligned64(void const *p) {
    return (reinterpret_cast<uintptr_t>(p) % 64) == 0;
}

} // namespace

TEST_CASE("MemoryPool carves aligned memory", "[memory][pool]") {
    MemoryPool pool(64 * kMiB, "align");

    std::vector<void *> carves;
    for (size_t const bytes : {1ULL, 7ULL, 64ULL, 4096ULL, 3ULL * kMiB, 7ULL * kMiB}) {
        void *p = pool.allocate(bytes);
        REQUIRE(p != nullptr);
        REQUIRE(aligned64(p));
        std::memset(p, 0xAB, bytes);
        carves.push_back(p);
    }

    REQUIRE(pool.bytes_used() >= 10 * kMiB);
    REQUIRE(pool.arenas() == 1);

    for (void *p : carves) {
        pool.deallocate(p);
    }
    REQUIRE(pool.bytes_used() == 0);
}

TEST_CASE("MemoryPool zero-size carves have distinct addresses", "[memory][pool]") {
    MemoryPool pool(kMiB, "zero");

    std::set<void *> seen;
    for (int i = 0; i < 64; i++) {
        void *p = pool.allocate(0);
        REQUIRE(p != nullptr);
        REQUIRE(aligned64(p));
        REQUIRE(seen.insert(p).second);
    }
}

TEST_CASE("MemoryPool reclaims freed bytes", "[memory][pool]") {
    // Sized for one 16 MiB block at a time. If freeing did not return the bytes
    // to the arena, the loop would exhaust it and force a second arena.
    MemoryPool pool(48 * kMiB, "reclaim");

    for (int i = 0; i < 64; i++) {
        void *p = pool.allocate(16 * kMiB);
        REQUIRE(p != nullptr);
        std::memset(p, i, 16 * kMiB);
        pool.deallocate(p);
    }

    REQUIRE(pool.arenas() == 1);
    REQUIRE(pool.bytes_used() == 0);
    REQUIRE(pool.high_water() >= 16 * kMiB);
}

TEST_CASE("MemoryPool grows when an exclusive arena runs out", "[memory][pool]") {
    // mimalloc floors an arena at mi_arena_min_size (32 MiB), so a nominally
    // tiny reserve is really one minimum arena. Holding far more than that live
    // is the documented overflow path: a null carve, a warning, another arena.
    MemoryPool pool(1, "overflow");
    REQUIRE(pool.arenas() == 1);
    size_t const first_arena = pool.bytes_reserved();

    std::vector<void *> held;
    for (int i = 0; i < 24; i++) {
        void *p = pool.allocate(8 * kMiB);
        REQUIRE(p != nullptr);
        REQUIRE(aligned64(p));
        std::memset(p, i, 8 * kMiB);
        held.push_back(p);
    }

    REQUIRE(pool.arenas() > 1);
    REQUIRE(pool.bytes_reserved() > first_arena);

    // Every block is still readable after the growth, so growing never moved
    // or invalidated an earlier carve.
    for (size_t i = 0; i < held.size(); i++) {
        REQUIRE(static_cast<unsigned char *>(held[i])[0] == static_cast<unsigned char>(i));
        pool.deallocate(held[i]);
    }
    REQUIRE(pool.bytes_used() == 0);
}

TEST_CASE("MemoryPool reserve grows capacity up front", "[memory][pool]") {
    MemoryPool   pool(kMiB, "reserve");
    size_t const before = pool.bytes_reserved();

    pool.reserve(before / 2);
    REQUIRE(pool.bytes_reserved() == before);

    pool.reserve(before + 128 * kMiB);
    REQUIRE(pool.bytes_reserved() >= before + 128 * kMiB);
    REQUIRE(pool.arenas() == 2);
}

TEST_CASE("MemoryPool reset refuses to run under live borrowers", "[memory][pool]") {
    MemoryPool pool(32 * kMiB, "reset");

    auto [ptr, token] = pool.allocate_borrowed(4 * kMiB);
    REQUIRE(ptr != nullptr);
    REQUIRE(pool.live_borrows() == 1);
    REQUIRE_THROWS_AS(pool.reset(), std::runtime_error);

    token.reset();
    REQUIRE(pool.live_borrows() == 0);
    REQUIRE(pool.bytes_used() == 0);

    // Untracked carves survive a release-free path but not a reset.
    void *raw = pool.allocate(4 * kMiB);
    REQUIRE(raw != nullptr);
    REQUIRE(pool.bytes_used() >= 4 * kMiB);
    REQUIRE_NOTHROW(pool.reset());
    REQUIRE(pool.bytes_used() == 0);

    // The pool is usable again after a reset.
    REQUIRE(pool.allocate(kMiB) != nullptr);
}

TEST_CASE("MemoryPool borrow tokens free on the last drop", "[memory][pool]") {
    MemoryPool pool(32 * kMiB, "borrow");

    {
        auto [ptr, token] = pool.allocate_borrowed(2 * kMiB);
        auto copy         = token;
        REQUIRE(pool.live_borrows() == 1);
        token.reset();
        REQUIRE(pool.live_borrows() == 1);
        REQUIRE(pool.bytes_used() >= 2 * kMiB);
    }

    REQUIRE(pool.live_borrows() == 0);
    REQUIRE(pool.bytes_used() == 0);
}

TEST_CASE("MemoryPool epochs nest and free their cohort", "[memory][pool]") {
    MemoryPool pool(64 * kMiB, "epochs");

    void *outer = pool.allocate(kMiB);
    REQUIRE(outer != nullptr);
    size_t const outer_used = pool.bytes_used();

    {
        auto first = pool.epoch();
        REQUIRE(pool.epoch_depth() == 1);
        REQUIRE(first.is_open());
        REQUIRE(pool.allocate(2 * kMiB) != nullptr);

        {
            auto second = pool.epoch();
            REQUIRE(pool.epoch_depth() == 2);
            REQUIRE(pool.allocate(4 * kMiB) != nullptr);
        }

        REQUIRE(pool.epoch_depth() == 1);
        // The inner cohort is gone even though nothing tracked it; the outer
        // epoch's own carve is not.
        REQUIRE(pool.bytes_used() >= outer_used + 2 * kMiB);
        REQUIRE(pool.bytes_used() < outer_used + 6 * kMiB);

        first.close();
        REQUIRE_FALSE(first.is_open());
        REQUIRE(pool.epoch_depth() == 0);
    }

    REQUIRE(pool.bytes_used() == outer_used);
    pool.deallocate(outer);
    REQUIRE(pool.bytes_used() == 0);
}

TEST_CASE("MemoryPool epoch close refuses to run under live borrowers", "[memory][pool]") {
    MemoryPool pool(32 * kMiB, "epoch-borrow");

    auto                        scope = pool.epoch();
    std::shared_ptr<void const> token;
    {
        auto [ptr, tok] = pool.allocate_borrowed(kMiB);
        token           = tok;
        REQUIRE(ptr != nullptr);
    }

    REQUIRE_THROWS_AS(scope.close(), std::runtime_error);
    REQUIRE(scope.is_open());

    token.reset();
    REQUIRE_NOTHROW(scope.close());
    REQUIRE(pool.epoch_depth() == 0);
    REQUIRE(pool.bytes_used() == 0);
}

TEST_CASE("MemoryPool carves outlive the pool handle", "[memory][pool]") {
    std::shared_ptr<void const> token;
    void                       *raw = nullptr;

    {
        auto pool       = MemoryPool::create(32 * kMiB, "outlive");
        auto [ptr, tok] = pool->allocate_borrowed(kMiB);
        raw             = ptr;
        token           = tok;
        std::memset(raw, 0x5A, kMiB);
    }

    // The pool object is gone; the arena is not, because the token holds it.
    REQUIRE(static_cast<unsigned char *>(raw)[0] == 0x5A);
    REQUIRE(static_cast<unsigned char *>(raw)[kMiB - 1] == 0x5A);
    token.reset();
}

TEST_CASE("MemoryPool carves are freed from other threads", "[memory][pool]") {
    MemoryPool pool(32 * kMiB, "cross-thread");

    auto [ptr, token] = pool.allocate_borrowed(8 * kMiB);
    std::memset(ptr, 0x11, 8 * kMiB);
    REQUIRE(pool.live_borrows() == 1);

    std::thread killer([tok = std::move(token)]() mutable { tok.reset(); });
    killer.join();

    REQUIRE(pool.live_borrows() == 0);
    REQUIRE(pool.bytes_used() == 0);

    // The bytes really came back: the pool serves the same size again without
    // reaching for another arena.
    REQUIRE(pool.allocate(8 * kMiB) != nullptr);
    REQUIRE(pool.arenas() == 1);
}

TEST_CASE("MemoryPool refuses to carve off the owning thread", "[memory][pool]") {
    MemoryPool pool(32 * kMiB, "affinity");

    bool        threw = false;
    std::thread other([&]() {
        try {
            (void)pool.allocate(kMiB);
        } catch (std::runtime_error const &) {
            threw = true;
        }
    });
    other.join();

    REQUIRE(threw);
}

TEST_CASE("MemoryPool reports its accounting", "[memory][pool]") {
    MemoryPool pool(64 * kMiB, "stats", 8 * kMiB);

    REQUIRE(pool.name() == "stats");
    REQUIRE(pool.warn_bytes() == 8 * kMiB);
    REQUIRE(pool.bytes_reserved() >= 64 * kMiB);
    REQUIRE(pool.bytes_used() == 0);
    REQUIRE(pool.high_water() == 0);

    void *a = pool.allocate(4 * kMiB);
    void *b = pool.allocate(4 * kMiB);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);

    auto const peak = pool.stats();
    REQUIRE(peak.bytes_used >= 8 * kMiB);
    REQUIRE(peak.high_water == peak.bytes_used);
    REQUIRE(peak.arenas == 1);
    REQUIRE(peak.epoch_depth == 0);

    pool.deallocate(a);
    pool.deallocate(b);

    auto const after = pool.stats();
    REQUIRE(after.bytes_used == 0);
    REQUIRE(after.high_water == peak.high_water);

    pool.set_warn_bytes(0);
    REQUIRE(pool.warn_bytes() == 0);
}

TEST_CASE("MemoryPool enforces the einsums:max-memory planning ceiling", "[memory][pool]") {
    // The counter is process-monotonic, so the ceiling is set relative to
    // whatever earlier tests already reserved.
    std::string const saved = config::get(option::MaxMemory);

    size_t const already = pooled_reserved_bytes();
    config::set(option::MaxMemory, std::to_string((already + 64 * kMiB) / kMiB) + "MB");

    // Within the ceiling: a small pool still fits (32 MiB arena minimum).
    REQUIRE_NOTHROW(MemoryPool(1 * kMiB, "under-ceiling"));

    // Past it: refused at the reserve site with the option named.
    REQUIRE_THROWS_AS(MemoryPool(256 * kMiB, "over-ceiling"), std::runtime_error);

    // Disabled: the same reservation is allowed again.
    config::set(option::MaxMemory, std::string("0"));
    REQUIRE_NOTHROW(MemoryPool(256 * kMiB, "no-ceiling"));

    config::set(option::MaxMemory, saved);
}
