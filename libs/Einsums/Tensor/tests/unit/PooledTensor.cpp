//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/PooledTensor.hpp>

#include <cstdint>
#include <thread>

#include <Einsums/Testing.hpp>

using namespace einsums;

namespace {

constexpr size_t kMiB = 1024ULL * 1024ULL;

bool aligned64(void const *p) {
    return (reinterpret_cast<uintptr_t>(p) % 64) == 0;
}

} // namespace

TEMPLATE_TEST_CASE("pooled runtime tensors carve from the pool", "[tensor][pool]", float, double, std::complex<float>,
                   std::complex<double>) {
    MemoryPool pool(64 * kMiB, "runtime");

    auto t = pool_empty<TestType>(pool, "A", {40, 50, 6});

    REQUIRE(t.name() == "A");
    REQUIRE(t.rank() == 3);
    REQUIRE(t.dim(0) == 40);
    REQUIRE(t.dim(2) == 6);
    REQUIRE(t.size() == 40 * 50 * 6);
    REQUIRE(t.data() != nullptr);
    REQUIRE(aligned64(t.data()));

    // The carve survived the return. A deep copy on the way out (which is what
    // a compiler that skipped NRVO would do, these types having no move
    // constructor) would have dropped the token and left nothing borrowed.
    REQUIRE(pool.live_borrows() == 1);
    REQUIRE(pool.bytes_used() >= t.size() * sizeof(TestType));

    // Writable, and readable back.
    for (size_t i = 0; i < t.size(); i++) {
        t.data()[i] = static_cast<TestType>(static_cast<double>(i % 17));
    }
    REQUIRE(t.data()[16] == static_cast<TestType>(16.0));
}

TEMPLATE_TEST_CASE("pooled zeros are zero", "[tensor][pool]", float, double, std::complex<float>, std::complex<double>) {
    MemoryPool pool(32 * kMiB, "zeros");

    // Dirty the arena first so a zeroed tensor cannot pass by accident.
    {
        auto dirty = pool_empty<TestType>(pool, "dirty", {10000});
        for (size_t i = 0; i < dirty.size(); i++) {
            dirty.data()[i] = static_cast<TestType>(3.5);
        }
    }

    auto z = pool_zeros<TestType>(pool, "Z", {10000});
    for (size_t i = 0; i < z.size(); i++) {
        REQUIRE(z.data()[i] == TestType{0});
    }
}

TEST_CASE("pooled compile-time-rank tensors carve from the pool", "[tensor][pool]") {
    MemoryPool pool(32 * kMiB, "ranked");

    auto t = pool_tensor<double, 2>(pool, "M", 64, 64);
    REQUIRE(t.dim(0) == 64);
    REQUIRE(t.dim(1) == 64);
    REQUIRE(aligned64(t.data()));
    REQUIRE(pool.live_borrows() == 1);

    auto z = pool_zero_tensor<double, 3>(pool, "Z", 8, 8, 8);
    REQUIRE(z.size() == 512);
    for (size_t i = 0; i < z.size(); i++) {
        REQUIRE(z.data()[i] == 0.0);
    }
    REQUIRE(pool.live_borrows() == 2);
}

TEST_CASE("a pooled tensor's death returns its bytes", "[tensor][pool]") {
    MemoryPool pool(48 * kMiB, "lifetime");

    // One 16 MiB tensor at a time, many times over: only real reclamation keeps
    // this inside a single arena.
    for (int i = 0; i < 32; i++) {
        auto t = pool_empty<double>(pool, "scratch", {2 * kMiB});
        REQUIRE(t.size() * sizeof(double) == 16 * kMiB);
        t.data()[0] = static_cast<double>(i);
    }

    REQUIRE(pool.arenas() == 1);
    REQUIRE(pool.live_borrows() == 0);
    REQUIRE(pool.bytes_used() == 0);
}

TEST_CASE("a pooled tensor outlives its pool", "[tensor][pool]") {
    // Built on the heap so the tensor, and with it the keepalive token holding
    // the arena open, can survive the scope that owns the pool.
    RuntimeTensor<double> *held = nullptr;

    {
        MemoryPool pool(32 * kMiB, "outlive");
        held = new RuntimeTensor<double>(RuntimeTensor<double>::deferred_alloc, "kept", std::vector<size_t>{1024, 128});
        pool.place(*held);
        for (size_t i = 0; i < held->size(); i++) {
            held->data()[i] = 2.0;
        }
    }

    REQUIRE(held->data()[0] == 2.0);
    REQUIRE(held->data()[held->size() - 1] == 2.0);
    delete held;
}

TEST_CASE("a pooled tensor may die on another thread", "[tensor][pool]") {
    MemoryPool pool(32 * kMiB, "cross-thread");

    auto *t = new RuntimeTensor<double>(RuntimeTensor<double>::deferred_alloc, "far", std::vector<size_t>{1024, 512});
    pool.place(*t);
    t->data()[0] = 1.0;
    REQUIRE(pool.live_borrows() == 1);

    std::thread killer([t]() { delete t; });
    killer.join();

    REQUIRE(pool.live_borrows() == 0);
    REQUIRE(pool.bytes_used() == 0);
}

TEST_CASE("resizing a pooled tensor returns the carve", "[tensor][pool]") {
    MemoryPool pool(32 * kMiB, "resize");

    auto t = pool_empty<double>(pool, "R", {256, 256});
    REQUIRE(pool.live_borrows() == 1);

    t.resize(std::vector<size_t>{128, 128});
    REQUIRE(t.size() == 128 * 128);
    REQUIRE(pool.live_borrows() == 0);
    REQUIRE(pool.bytes_used() == 0);
    // The tensor is self-owned now and still works.
    t.data()[0] = 4.0;
    REQUIRE(t.data()[0] == 4.0);
}

TEST_CASE("releasing a pooled tensor returns the carve", "[tensor][pool]") {
    MemoryPool pool(32 * kMiB, "release");

    auto t = pool_empty<double>(pool, "R", {256, 256});
    REQUIRE(t.is_materialized());
    REQUIRE(pool.live_borrows() == 1);

    t.release();
    REQUIRE_FALSE(t.is_materialized());
    REQUIRE(pool.live_borrows() == 0);
    REQUIRE(pool.bytes_used() == 0);
}

TEST_CASE("copying a pooled tensor takes it off the pool", "[tensor][pool]") {
    MemoryPool pool(32 * kMiB, "copy");

    auto t = pool_empty<double>(pool, "src", {64, 64});
    for (size_t i = 0; i < t.size(); i++) {
        t.data()[i] = static_cast<double>(i);
    }

    RuntimeTensor<double> copy = t;
    REQUIRE(copy.data() != t.data());
    REQUIRE(copy.data()[5] == 5.0);
    REQUIRE(pool.live_borrows() == 1); // the copy owns its own buffer

    copy.data()[5] = -1.0;
    REQUIRE(t.data()[5] == 5.0);
}

TEST_CASE("pooled tensors within an epoch die with it", "[tensor][pool]") {
    MemoryPool pool(64 * kMiB, "epoch");

    auto         keep    = pool_empty<double>(pool, "keep", {1024});
    size_t const outside = pool.bytes_used();

    {
        auto scope = pool.epoch();
        {
            auto inner = pool_empty<double>(pool, "inner", {1024, 128});
            REQUIRE(pool.bytes_used() > outside);
        }
        REQUIRE(pool.bytes_used() == outside);
        scope.close();
    }

    REQUIRE(pool.bytes_used() == outside);
    REQUIRE(pool.live_borrows() == 1);
}

TEST_CASE("an epoch refuses to close under a live pooled tensor", "[tensor][pool]") {
    MemoryPool pool(32 * kMiB, "epoch-live");

    auto scope = pool.epoch();
    auto held  = pool_empty<double>(pool, "held", {1024});

    REQUIRE_THROWS_AS(scope.close(), std::runtime_error);
    REQUIRE(scope.is_open());

    held.release();
    REQUIRE_NOTHROW(scope.close());
}
