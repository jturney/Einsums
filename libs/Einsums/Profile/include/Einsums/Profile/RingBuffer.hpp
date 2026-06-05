//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <atomic>
#    include <cstddef>
#    include <new>

namespace einsums::profile {

/// Single-Producer, Single-Consumer (SPSC) ring buffer.
/// The producer (application thread) writes events; the consumer thread reads them.
/// Never blocks the producer — try_push() returns false if the buffer is full.
///
/// Template parameters:
///   T        — element type (must be trivially copyable)
///   Capacity — must be a power of 2
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
    auto try_push(T const &item) -> bool {
        size_t const h    = _head.load(std::memory_order_relaxed);
        size_t const next = (h + 1) & mask_;
        if (next == _tail.load(std::memory_order_acquire))
            return false; // full
        _buffer[h & mask_] = item;
        _head.store(next, std::memory_order_release);
        return true;
    }

    /// Try to pop an element. Returns false if buffer is empty.
    auto try_pop(T &item) -> bool {
        size_t const t = _tail.load(std::memory_order_relaxed);
        if (t == _head.load(std::memory_order_acquire))
            return false; // empty
        item = _buffer[t & mask_];
        _tail.store((t + 1) & mask_, std::memory_order_release);
        return true;
    }

    /// Check if buffer is empty (approximate — may race with producer).
    auto empty() const -> bool { return _tail.load(std::memory_order_acquire) == _head.load(std::memory_order_acquire); }

  private:
    static constexpr size_t mask_ = Capacity - 1;

    // Cache-line padding to prevent false sharing between head and tail
#    ifdef __cpp_lib_hardware_interference_size
    static constexpr size_t cache_line = std::hardware_destructive_interference_size;
#    else
    static constexpr size_t cache_line = 64;
#    endif

    alignas(cache_line) std::atomic<size_t> _head{0};
    alignas(cache_line) std::atomic<size_t> _tail{0};

    T _buffer[Capacity]; // NOLINT(modernize-avoid-c-arrays)
};

} // namespace einsums::profile

#endif
