//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// Lock-free work-stealing deque based on Chase & Lev (SPAA 2005).
// Owner thread pushes/pops from the bottom (LIFO).
// Thief threads steal from the top (FIFO).

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace einsums::task_pool {

/// @brief Lock-free single-producer, multi-consumer work-stealing deque.
///
/// The owner thread calls push() and pop() (bottom end).
/// Any thread can call steal() (top end).
///
/// Based on the Chase-Lev dynamic circular work-stealing deque.
/// Uses a growable circular buffer with atomic indices.
template <typename T>
class WorkStealingDeque {
  public:
    explicit WorkStealingDeque(size_t initial_capacity = 1024)
        : _buffer(std::make_unique<CircularBuffer>(initial_capacity)), _top(0), _bottom(0) {}

    WorkStealingDeque(WorkStealingDeque const &)            = delete;
    WorkStealingDeque &operator=(WorkStealingDeque const &) = delete;

    /// @brief Push an item onto the bottom (owner thread only).
    void push(T item) {
        int64_t b   = _bottom.load(std::memory_order_relaxed);
        int64_t t   = _top.load(std::memory_order_acquire);
        auto   *buf = _buffer.get();

        if (b - t >= static_cast<int64_t>(buf->capacity())) {
            // Grow
            buf = grow(buf, b, t);
        }

        buf->store(b, std::move(item));
        // Release ensures the item is visible before bottom_ update is visible.
        _bottom.store(b + 1, std::memory_order_release);
    }

    /// @brief Pop an item from the bottom (owner thread only, LIFO).
    /// @return The item, or nullopt if empty.
    std::optional<T> pop() {
        int64_t b = _bottom.load(std::memory_order_relaxed) - 1;
        _bottom.store(b, std::memory_order_relaxed);

        auto *buf = _buffer.get();
        std::atomic_thread_fence(std::memory_order_seq_cst);

        int64_t t = _top.load(std::memory_order_relaxed);

        if (t <= b) {
            // Non-empty
            T item = buf->load(b);
            if (t == b) {
                // Last element; race with steal()
                if (!_top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                    // Lost race
                    _bottom.store(b + 1, std::memory_order_relaxed);
                    return std::nullopt;
                }
                _bottom.store(b + 1, std::memory_order_relaxed);
            }
            return item;
        }

        // Empty
        _bottom.store(b + 1, std::memory_order_relaxed);
        return std::nullopt;
    }

    /// @brief Steal an item from the top (any thread, FIFO).
    /// @return The item, or nullopt if empty or contended.
    std::optional<T> steal() {
        int64_t t = _top.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t b = _bottom.load(std::memory_order_acquire);

        if (t < b) {
            auto *buf  = _buffer.get();
            T     item = buf->load(t);

            if (!_top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                return std::nullopt; // Lost race with another steal or pop
            }
            return item;
        }

        return std::nullopt; // Empty
    }

    /// @brief Approximate size (may be stale).
    [[nodiscard]] size_t size_approx() const {
        int64_t b = _bottom.load(std::memory_order_relaxed);
        int64_t t = _top.load(std::memory_order_relaxed);
        return static_cast<size_t>(std::max(int64_t{0}, b - t));
    }

    /// @brief Check if approximately empty.
    [[nodiscard]] bool empty_approx() const { return size_approx() == 0; }

  private:
    struct CircularBuffer {
        explicit CircularBuffer(size_t cap) : data_(cap), mask_(cap - 1) {
            // cap must be power of 2
        }

        [[nodiscard]] size_t capacity() const { return data_.size(); }

        void store(int64_t idx, T item) { data_[static_cast<size_t>(idx) & mask_] = std::move(item); }

        T load(int64_t idx) const { return data_[static_cast<size_t>(idx) & mask_]; }

        std::vector<T> data_;
        size_t         mask_;
    };

    CircularBuffer *grow(CircularBuffer *old, int64_t b, int64_t t) {
        size_t new_cap = old->capacity() * 2;
        auto   new_buf = std::make_unique<CircularBuffer>(new_cap);
        for (int64_t i = t; i < b; i++) {
            new_buf->store(i, old->load(i));
        }
        auto *raw = new_buf.get();
        // Keep old buffers alive to prevent use-after-free from concurrent steal()
        _old_buffers.push_back(std::move(_buffer));
        _buffer = std::move(new_buf);
        return raw;
    }

    std::unique_ptr<CircularBuffer>              _buffer;
    std::vector<std::unique_ptr<CircularBuffer>> _old_buffers; // Keeps old buffers alive
    std::atomic<int64_t>                         _top;
    std::atomic<int64_t>                         _bottom;
};

} // namespace einsums::task_pool
