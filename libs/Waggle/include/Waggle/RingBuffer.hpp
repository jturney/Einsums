//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Waggle/Config.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>

WAGGLE_NAMESPACE_BEGIN

/// Single-producer, single-consumer ring buffer that never blocks the producer.
///
/// The indices run free and are masked only to address a slot, so every slot is usable and the
/// fill level is a plain difference.
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

    /// Producer side: the next slot to fill in place, or nullptr (counted in @ref refused) if the
    /// buffer is full. @ref commit publishes it. Reads the consumer's tail only when the cached copy
    /// says the buffer is full.
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

    /// Producer side: whether the buffer is more than half full. Reads the consumer's tail only when
    /// the cached copy, which can only overstate the fill, says so.
    auto past_half() -> bool {
        size_t const h = _head.load(std::memory_order_relaxed);
        if (h - _cached_tail <= Capacity / 2) {
            return false;
        }
        _cached_tail = _tail.load(std::memory_order_acquire);
        return h - _cached_tail > Capacity / 2;
    }

    /// Pushes refused because the buffer was full. Kept on the producer's cache line: drops come
    /// from every thread at once, and a shared count would bounce between their cores.
    auto refused() const -> uint64_t { return _refused.load(std::memory_order_relaxed); }

    /// Try to pop an element; false if the buffer is empty. One consumer at a time: callers draining
    /// from several threads must serialize.
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

    /// Consumer side: call @p f on each element in place until the buffer is empty; returns the count.
    /// Same one-consumer rule as @ref try_pop. The tail is published every @ref kReleaseEvery
    /// elements, not after each, so a busy drain does not make every producer read of it a miss.
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

    // A fixed 64, not std::hardware_destructive_interference_size, which follows -mtune: this layout
    // is ABI, read by inline code in every TU that records a zone.
    static constexpr size_t cache_line = 64;

    // Each side's index shares a line with its cached copy of the other's, so a push touches only
    // the producer's line and a pop only the consumer's.
    alignas(cache_line) std::atomic<size_t> _head{0};
    size_t                _cached_tail{0}; // the producer's view of _tail
    std::atomic<uint64_t> _refused{0};
    alignas(cache_line) std::atomic<size_t> _tail{0};
    size_t _cached_head{0}; // the consumer's view of _head

    // On a line of its own, so the first slot does not share the consumer's line for any T.
    alignas(cache_line) T _buffer[Capacity]; // NOLINT(modernize-avoid-c-arrays)
};

WAGGLE_NAMESPACE_END
