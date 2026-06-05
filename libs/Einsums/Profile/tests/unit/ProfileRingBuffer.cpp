//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Profile/Event.hpp>
#include <Einsums/Profile/RingBuffer.hpp>

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <vector>

#if defined(EINSUMS_HAVE_PROFILER)

using namespace einsums::profile;

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

    // Can push 7 items (capacity - 1 usable slots in SPSC)
    for (int i = 0; i < 7; ++i) {
        REQUIRE(rb.try_push(i));
    }

    // 8th push should fail (buffer full)
    REQUIRE_FALSE(rb.try_push(99));

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
    push_evt.type      = EventType::Push;
    push_evt.timestamp = Clock::now();
    push_evt.name_id   = 42;
    push_evt.file_id   = 1;
    push_evt.func_id   = 2;
    push_evt.line      = 100;

    REQUIRE(rb.try_push(push_evt));

    Event pop_evt{};
    REQUIRE(rb.try_pop(pop_evt));
    REQUIRE(pop_evt.type == EventType::Push);
    REQUIRE(pop_evt.name_id == 42);
    REQUIRE(pop_evt.line == 100);
}

#endif
