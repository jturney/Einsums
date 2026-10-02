//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <atomic>
#    include <cstddef>
#    include <cstdint>

EINSUMS_NAMESPACE_BEGIN(profile)

/// Single-Producer, Single-Consumer (SPSC) ring buffer.
///
/// The producer (application thread) writes events; the consumer thread reads them.
/// Never blocks the producer; try_push() returns false if the buffer is full.
///
/// The indices run free and are masked only to address a slot, so all @p Capacity slots are usable
/// and the fill level is a plain difference. A size_t does not wrap in the life of a process.
///
/// @tparam T The element type, which must be trivially copyable.
/// @tparam Capacity The number of slots, which must be a power of 2.
template <typename T, size_t Capacity>
class RingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static_assert(Capacity > 0);

  public:
    RingBuffer() = default;

    // Non-copyable, non-movable
    RingBuffer(RingBuffer const &)            = delete;
    RingBuffer &operator=(RingBuffer const &) = delete;

    /// Producer side: the next slot to write, or nullptr if the buffer is full. A null claim is
    /// counted in @ref refused. The slot is published by @ref commit, so the producer fills it in
    /// place instead of building the element elsewhere and copying it in.
    ///
    /// Reads the consumer's tail only when the producer's cached copy says the buffer is full: the
    /// tail is the line the consumer writes as it drains, so reading it on every push missed the
    /// cache whenever the consumer had just been through.
    auto try_claim() -> T * {
        size_t const h = _head.load(std::memory_order_relaxed);
        if (h - _cached_tail == Capacity) {
            _cached_tail = _tail.load(std::memory_order_acquire);
            if (h - _cached_tail == Capacity) {
                // Only this thread writes the count, so a load and store, not a read-modify-write.
                _refused.store(_refused.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
                return nullptr;
            }
        }
        return &_buffer[h & mask_];
    }

    /// Producer side: publish the slot the last successful @ref try_claim returned.
    void commit() { _head.store(_head.load(std::memory_order_relaxed) + 1, std::memory_order_release); }

    /// Try to push an element. Returns false if buffer is full (never blocks).
    auto try_push(T const &item) -> bool {
        T *slot = try_claim();
        if (slot == nullptr) {
            return false;
        }
        *slot = item;
        commit();
        return true;
    }

    /// Producer side: whether the buffer holds more than half its capacity. Reads the consumer's tail
    /// only when the cached copy says so, and the cached copy only ever overstates what is held.
    auto past_half() -> bool {
        size_t const h = _head.load(std::memory_order_relaxed);
        if (h - _cached_tail <= Capacity / 2) {
            return false;
        }
        _cached_tail = _tail.load(std::memory_order_acquire);
        return h - _cached_tail > Capacity / 2;
    }

    /// Pushes refused because the buffer was full. Written only by the producer, on its own cache
    /// line: a count every producer shared bounced between their cores on every drop, and drops come
    /// in floods, from every thread at once, exactly when the rings are overflowing.
    auto refused() const -> uint64_t { return _refused.load(std::memory_order_relaxed); }

    /// Try to pop an element. Returns false if buffer is empty.
    ///
    /// Reads the producer's head only when the consumer's cached copy says the buffer is empty. One
    /// consumer at a time: callers that drain from more than one thread must serialize themselves.
    auto try_pop(T &item) -> bool {
        size_t const t = _tail.load(std::memory_order_relaxed);
        if (t == _cached_head) {
            _cached_head = _head.load(std::memory_order_acquire);
            if (t == _cached_head)
                return false; // empty
        }
        item = _buffer[t & mask_];
        _tail.store(t + 1, std::memory_order_release);
        return true;
    }

    /// Consumer side: call @p f on every element until the buffer is empty, in place, and return how
    /// many there were. Same one-consumer rule as @ref try_pop.
    ///
    /// The tail is published every @ref kReleaseEvery elements rather than after each. The producer
    /// reads the tail whenever its ring is full or past half, which is when the consumer is busiest,
    /// and a tail written on every pop made each of those reads a miss. Releasing in steps still
    /// hands space back long before a 64Ki-slot ring would notice it was held.
    template <typename F>
    auto drain(F &&f) -> size_t {
        size_t const first = _tail.load(std::memory_order_relaxed);
        size_t       t     = first;
        while (true) {
            _cached_head = _head.load(std::memory_order_acquire);
            if (t == _cached_head) {
                break;
            }
            while (t != _cached_head) {
                f(static_cast<T const &>(_buffer[t & mask_]));
                ++t;
                if ((t & (kReleaseEvery - 1)) == 0) {
                    _tail.store(t, std::memory_order_release);
                }
            }
            _tail.store(t, std::memory_order_release);
        }
        return t - first;
    }

    /// Check if buffer is empty. The result is approximate and may race with the producer.
    auto empty() const -> bool { return _tail.load(std::memory_order_acquire) == _head.load(std::memory_order_acquire); }

  private:
    static constexpr size_t mask_         = Capacity - 1;
    static constexpr size_t kReleaseEvery = Capacity < 256 ? Capacity : 256;

    // Cache-line padding to prevent false sharing between head and tail. A fixed 64, not
    // std::hardware_destructive_interference_size: that value follows -mtune and -mcpu, and this
    // layout is part of the ABI, reached by inline code in every TU that records a zone, so a
    // program tuned differently from the library would disagree with it about where _tail is.
    static constexpr size_t cache_line = 64;

    // Each side's index shares a line with its cached copy of the other's, so a push touches only
    // the producer's line and a pop only the consumer's, until a cached copy runs out.
    alignas(cache_line) std::atomic<size_t> _head{0};
    size_t                _cached_tail{0}; // the producer's view of _tail
    std::atomic<uint64_t> _refused{0};
    alignas(cache_line) std::atomic<size_t> _tail{0};
    size_t _cached_head{0}; // the consumer's view of _head

    // On a line of its own, so the first slot does not share the consumer's line for any T.
    alignas(cache_line) T _buffer[Capacity]; // NOLINT(modernize-avoid-c-arrays)
};

EINSUMS_NAMESPACE_END(profile)

#endif
