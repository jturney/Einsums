//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Waggle/Clock.hpp>
#include <Waggle/Event.hpp>
#include <Waggle/RingBuffer.hpp>

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <vector>

using namespace waggle;

TEST_CASE("RingBuffer basic push/pop", "[profiler][ringbuffer]") {
    RingBuffer<int, 16> rb;

    REQUIRE(rb.empty());

    int val = 0;
    REQUIRE_FALSE(rb.try_pop(val));

    REQUIRE(rb.try_push(42));
    REQUIRE_FALSE(rb.empty());

    REQUIRE(rb.try_pop(val));
    REQUIRE(val == 42);
    REQUIRE(rb.empty());
}

TEST_CASE("RingBuffer fills to capacity", "[profiler][ringbuffer]") {
    RingBuffer<int, 8> rb;

    // Every slot is usable: the indices run free, so a full ring is not mistaken for an empty one
    for (int i = 0; i < 8; ++i) {
        REQUIRE(rb.try_push(i));
    }

    // 9th push should fail (buffer full)
    REQUIRE_FALSE(rb.try_push(99));
    REQUIRE(rb.refused() == 1);

    // Pop one and try again
    int val = 0;
    REQUIRE(rb.try_pop(val));
    REQUIRE(val == 0);

    REQUIRE(rb.try_push(99));
}

TEST_CASE("RingBuffer FIFO order", "[profiler][ringbuffer]") {
    RingBuffer<int, 64> rb;

    for (int i = 0; i < 50; ++i) {
        REQUIRE(rb.try_push(i));
    }

    for (int i = 0; i < 50; ++i) {
        int val = -1;
        REQUIRE(rb.try_pop(val));
        REQUIRE(val == i);
    }
}

TEST_CASE("RingBuffer concurrent producer/consumer", "[profiler][ringbuffer]") {
    RingBuffer<int, 65536> rb;
    constexpr int          count = 100000;

    std::atomic<bool> done{false};
    std::vector<int>  received;
    received.reserve(count);

    // Consumer thread
    std::thread consumer([&] {
        int val;
        int expected = 0;
        while (expected < count) {
            if (rb.try_pop(val)) {
                received.push_back(val);
                ++expected;
            }
        }
    });

    // Producer (this thread)
    int pushed = 0;
    while (pushed < count) {
        if (rb.try_push(pushed)) {
            ++pushed;
        }
    }

    consumer.join();

    REQUIRE(received.size() == count);
    for (int i = 0; i < count; ++i) {
        REQUIRE(received[i] == i);
    }
}

TEST_CASE("RingBuffer with Event struct", "[profiler][ringbuffer]") {
    RingBuffer<Event, 64> rb;

    Event push_evt{};
    push_evt.type    = EventType::Push;
    push_evt.ticks   = TickClock::now();
    push_evt.site_id = 7;
    push_evt.name_id = 42;

    REQUIRE(rb.try_push(push_evt));

    Event pop_evt{};
    REQUIRE(rb.try_pop(pop_evt));
    REQUIRE(pop_evt.type == EventType::Push);
    REQUIRE(pop_evt.site_id == 7);
    REQUIRE(pop_evt.name_id == 42);
}

TEST_CASE("RingBuffer counts the pushes it refuses", "[profiler][ringbuffer]") {
    // The drop count lives in the ring, on the producer's line, so producers no longer share one.
    RingBuffer<int, 4> rb;
    for (int i = 0; i < 4; ++i) {
        REQUIRE(rb.try_push(i));
    }
    REQUIRE(rb.refused() == 0);
    for (int i = 0; i < 3; ++i) {
        REQUIRE_FALSE(rb.try_push(i));
        REQUIRE(rb.try_claim() == nullptr);
    }
    REQUIRE(rb.refused() == 6);
}

TEST_CASE("RingBuffer claim and commit", "[profiler][ringbuffer]") {
    RingBuffer<int, 8> rb;

    int *slot = rb.try_claim();
    REQUIRE(slot != nullptr);
    *slot = 7;
    // Claimed but not committed: the consumer must not see it yet.
    REQUIRE(rb.empty());
    int val = 0;
    REQUIRE_FALSE(rb.try_pop(val));

    rb.commit();
    REQUIRE(rb.try_pop(val));
    REQUIRE(val == 7);
}

TEST_CASE("RingBuffer drain visits every element in order", "[profiler][ringbuffer]") {
    RingBuffer<int, 1024> rb;
    // Wrap the indices past the end of the buffer first.
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < 1000; ++i) {
            REQUIRE(rb.try_push(i));
        }
        int  expected = 0;
        auto n        = rb.drain([&](int const &v) { REQUIRE(v == expected++); });
        REQUIRE(n == 1000);
        REQUIRE(rb.empty());
        REQUIRE(rb.drain([](int const &) { FAIL("drained an empty ring"); }) == 0);
    }
}

TEST_CASE("RingBuffer drain hands space back before it finishes", "[profiler][ringbuffer]") {
    // The tail is released in steps during a drain, not only at its end: a drain of a full ring
    // takes milliseconds, and a producer that saw no space for all of them would drop throughout.
    RingBuffer<int, 1024> rb;
    for (int i = 0; i < 1024; ++i) {
        REQUIRE(rb.try_push(i));
    }
    REQUIRE_FALSE(rb.try_push(-1));

    int  seen = 0;
    bool room = false;
    rb.drain([&](int const &) {
        if (++seen == 512) {
            room = rb.try_push(-1);
        }
    });
    REQUIRE(room);
    // The push made mid-drain was found by the same drain, after the 1024 before it.
    REQUIRE(seen == 1025);
}

TEST_CASE("RingBuffer concurrent producer and draining consumer", "[profiler][ringbuffer]") {
    RingBuffer<int, 1024> rb;
    constexpr int         count = 1000000;

    std::vector<int> received;
    received.reserve(count);
    std::thread consumer([&] {
        while (received.size() < count) {
            rb.drain([&](int const &v) { received.push_back(v); });
        }
    });

    for (int pushed = 0; pushed < count;) {
        if (int *slot = rb.try_claim()) {
            *slot = pushed++;
            rb.commit();
        }
    }
    consumer.join();

    REQUIRE(received.size() == count);
    for (int i = 0; i < count; ++i) {
        REQUIRE(received[i] == i);
    }
}
