//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The record of which copy of a tensor, host or device, is current during a replay on a build whose
// device memory is separate. Such a build has no CI leg, so the replay's device paths never run in
// testing; these cases pin the rules those paths are built from, with copies that count instead of
// move bytes.

#include <Einsums/ComputeGraph/Detail/DeviceCoherence.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include <Einsums/Testing.hpp>

using Coherence = einsums::compute_graph::detail::DeviceCoherence;
using State     = Coherence::State;

namespace {

/// A copy that counts how often it ran and fails when told to.
struct CountingCopy {
    std::size_t calls{0};
    bool        succeed{true};

    auto fn() {
        return [this]() {
            ++calls;
            return succeed;
        };
    }
};

} // namespace

TEST_CASE("DeviceCoherence - a tracked tensor starts each replay current on the host alone", "[ComputeGraph][DeviceCoherence]") {
    Coherence                                           coherence;
    std::vector<einsums::compute_graph::TensorId> const ids{1, 2, 2};
    coherence.reset(ids);
    CHECK(coherence.tracked(1));
    CHECK(coherence.tracked(2));
    CHECK_FALSE(coherence.tracked(3));
    CHECK(coherence.state(1) == State::HostOnly);
    CHECK(coherence.state(2) == State::HostOnly);
}

TEST_CASE("DeviceCoherence - copies only toward a stale side", "[ComputeGraph][DeviceCoherence]") {
    Coherence                                           coherence;
    std::vector<einsums::compute_graph::TensorId> const ids{7};
    coherence.reset(ids);
    CountingCopy up;
    CountingCopy down;

    // Host current alone: asking for the host copies nothing.
    CHECK(coherence.need_on_host(7, down.fn()));
    CHECK(down.calls == 0);

    // The device copy is stale, so it is uploaded once; a second ask finds it current.
    CHECK(coherence.need_on_device(7, up.fn()));
    CHECK(coherence.need_on_device(7, up.fn()));
    CHECK(up.calls == 1);
    CHECK(coherence.state(7) == State::Both);

    // Reading on the device left the host current: no copy down.
    CHECK(coherence.need_on_host(7, down.fn()));
    CHECK(down.calls == 0);

    // A device write makes the host copy stale, and the next host reader copies it down once.
    coherence.wrote_on_device(7);
    CHECK(coherence.state(7) == State::DeviceOnly);
    CHECK(coherence.need_on_device(7, up.fn()));
    CHECK(up.calls == 1);
    CHECK(coherence.need_on_host(7, down.fn()));
    CHECK(coherence.need_on_host(7, down.fn()));
    CHECK(down.calls == 1);
    CHECK(coherence.state(7) == State::Both);

    // A host write makes the device copy stale again.
    coherence.wrote_on_host(7);
    CHECK(coherence.state(7) == State::HostOnly);
    CHECK(coherence.need_on_device(7, up.fn()));
    CHECK(up.calls == 2);
}

TEST_CASE("DeviceCoherence - a copy that did not happen changes nothing", "[ComputeGraph][DeviceCoherence]") {
    Coherence                                           coherence;
    std::vector<einsums::compute_graph::TensorId> const ids{4};
    coherence.reset(ids);
    CountingCopy up{.calls = 0, .succeed = false};
    CHECK_FALSE(coherence.need_on_device(4, up.fn()));
    CHECK(coherence.state(4) == State::HostOnly);

    coherence.wrote_on_device(4);
    CountingCopy down{.calls = 0, .succeed = false};
    CHECK_FALSE(coherence.need_on_host(4, down.fn()));
    CHECK(coherence.state(4) == State::DeviceOnly);
}

TEST_CASE("DeviceCoherence - an untracked tensor never leaves the host", "[ComputeGraph][DeviceCoherence]") {
    Coherence                                           coherence;
    std::vector<einsums::compute_graph::TensorId> const ids{1};
    coherence.reset(ids);
    CountingCopy copy;
    CHECK(coherence.need_on_host(9, copy.fn()));
    CHECK_FALSE(coherence.need_on_device(9, copy.fn()));
    coherence.wrote_on_device(9);
    CHECK(coherence.state(9) == State::HostOnly);
    CHECK(copy.calls == 0);
}

TEST_CASE("DeviceCoherence - the final flush copies down exactly the device-only tensors", "[ComputeGraph][DeviceCoherence]") {
    Coherence                                           coherence;
    std::vector<einsums::compute_graph::TensorId> const ids{1, 2, 3};
    coherence.reset(ids);
    CountingCopy up;
    coherence.need_on_device(2, up.fn()); // both current
    coherence.wrote_on_device(3);         // device only

    std::vector<einsums::compute_graph::TensorId> flushed;
    coherence.flush([&](einsums::compute_graph::TensorId id) {
        flushed.push_back(id);
        return true;
    });
    CHECK(flushed == std::vector<einsums::compute_graph::TensorId>{3});
    CHECK(coherence.state(3) == State::Both);

    // Reset forgets the replay: everything starts host-only again.
    coherence.reset(ids);
    CHECK(coherence.state(3) == State::HostOnly);
}

// Readers of one tensor are not ordered by the graph's dependency edges, so a concurrent executor
// can run several at once. The copy toward a stale side must happen once, never two at a time, and
// every reader must find its side current when it returns. The copy sleeps to hold the window open
// long enough for the other threads to arrive inside it.
TEST_CASE("DeviceCoherence - concurrent readers copy a stale side exactly once", "[ComputeGraph][DeviceCoherence]") {
    constexpr int threads     = 8;
    bool const    toward_host = GENERATE(true, false);

    Coherence                                           coherence;
    std::vector<einsums::compute_graph::TensorId> const ids{5};
    coherence.reset(ids);
    if (toward_host) {
        coherence.wrote_on_device(5);
    }

    std::atomic<int>  copies{0};
    std::atomic<int>  inside{0};
    std::atomic<bool> overlapped{false};
    std::atomic<int>  current{0};
    auto const        copy = [&]() {
        if (inside.fetch_add(1) != 0) {
            overlapped = true;
        }
        ++copies;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        inside.fetch_sub(1);
        return true;
    };

    std::atomic<bool>        go{false};
    std::vector<std::thread> readers;
    readers.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        readers.emplace_back([&]() {
            while (!go.load()) {
                std::this_thread::yield();
            }
            bool const ok = toward_host ? coherence.need_on_host(5, copy) : coherence.need_on_device(5, copy);
            if (ok) {
                ++current;
            }
        });
    }
    go = true;
    for (auto &reader : readers) {
        reader.join();
    }

    CHECK(copies.load() == 1);
    CHECK_FALSE(overlapped.load());
    CHECK(current.load() == threads);
    CHECK(coherence.state(5) == State::Both);
}

// Once both copies agree, readers on either side copy nothing, however many run at once: a reader
// never makes the other side stale, which is what keeps a host reader from copying down onto a
// buffer another host reader is reading.
TEST_CASE("DeviceCoherence - concurrent readers of agreeing copies copy nothing", "[ComputeGraph][DeviceCoherence]") {
    Coherence                                           coherence;
    std::vector<einsums::compute_graph::TensorId> const ids{6};
    coherence.reset(ids);
    std::atomic<int> copies{0};
    auto const       copy = [&]() {
        ++copies;
        return true;
    };
    REQUIRE(coherence.need_on_device(6, copy));
    copies = 0;

    std::vector<std::thread> readers;
    readers.reserve(8);
    for (int t = 0; t < 8; ++t) {
        readers.emplace_back([&, t]() {
            for (int i = 0; i < 1000; ++i) {
                if ((t + i) % 2 == 0) {
                    coherence.need_on_host(6, copy);
                } else {
                    coherence.need_on_device(6, copy);
                }
            }
        });
    }
    for (auto &reader : readers) {
        reader.join();
    }
    CHECK(copies.load() == 0);
    CHECK(coherence.state(6) == State::Both);
}
