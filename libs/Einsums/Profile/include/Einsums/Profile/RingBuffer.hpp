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
#    include <new>

EINSUMS_NAMESPACE_BEGIN(profile)

/// Single-Producer, Single-Consumer (SPSC) ring buffer.
///
/// The producer (application thread) writes events; the consumer thread reads them.
/// Never blocks the producer; try_push() returns false if the buffer is full.
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

    /// Try to push an element. Returns false if buffer is full (never blocks).
    ///
    /// Reads the consumer's tail only when the producer's cached copy says the buffer is full: the
    /// tail is the line the consumer writes as it drains, so reading it on every push missed the
    /// cache whenever the consumer had just been through.
    auto try_push(T const &item) -> bool {
        size_t const h    = _head.load(std::memory_order_relaxed);
        size_t const next = (h + 1) & mask_;
        if (next == _cached_tail) {
            _cached_tail = _tail.load(std::memory_order_acquire);
            if (next == _cached_tail)
                return false; // full
        }
        _buffer[h & mask_] = item;
        _head.store(next, std::memory_order_release);
        return true;
    }

    /// Producer side: whether the buffer holds more than half its capacity. Reads the consumer's tail
    /// only when the cached copy says so, and the cached copy only ever overstates what is held.
    auto past_half() -> bool {
        size_t const h = _head.load(std::memory_order_relaxed);
        if (((h - _cached_tail) & mask_) <= Capacity / 2) {
            return false;
        }
        _cached_tail = _tail.load(std::memory_order_acquire);
        return ((h - _cached_tail) & mask_) > Capacity / 2;
    }

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
        _tail.store((t + 1) & mask_, std::memory_order_release);
        return true;
    }

    /// Check if buffer is empty. The result is approximate and may race with the producer.
    auto empty() const -> bool { return _tail.load(std::memory_order_acquire) == _head.load(std::memory_order_acquire); }

  private:
    static constexpr size_t mask_ = Capacity - 1;

    // Cache-line padding to prevent false sharing between head and tail
#    ifdef __cpp_lib_hardware_interference_size
    static constexpr size_t cache_line = std::hardware_destructive_interference_size;
#    else
    static constexpr size_t cache_line = 64;
#    endif

    // Each side's index shares a line with its cached copy of the other's, so a push touches only
    // the producer's line and a pop only the consumer's, until a cached copy runs out.
    alignas(cache_line) std::atomic<size_t> _head{0};
    size_t _cached_tail{0}; // the producer's view of _tail
    alignas(cache_line) std::atomic<size_t> _tail{0};
    size_t _cached_head{0}; // the consumer's view of _head

    T _buffer[Capacity]; // NOLINT(modernize-avoid-c-arrays)
};

EINSUMS_NAMESPACE_END(profile)

#endif
