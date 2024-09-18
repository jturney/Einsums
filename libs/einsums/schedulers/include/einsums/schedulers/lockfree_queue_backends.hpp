//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_CXX11_STD_ATOMIC_128BIT)
#    include <einsums/concurrency/deque.hpp>
#else
#    include <boost/lockfree/queue.hpp>
#endif

#include <einsums/allocator_support/aligned_allocator.hpp>

// Does not rely on CXX11_STD_ATOMIC_128BIT
#include <einsums/concurrency/concurrentqueue.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace einsums::threads::detail {

////////////////////////////////////////////////////////////////////////////
// FIFO
template <typename T>
struct lockfree_fifo_backend {
    using container_type = einsums::concurrency::detail::ConcurrentQueue<T>;

    using value_type       = T;
    using reference        = T &;
    using const_reference  = T const &;
    using rvalue_reference = T &&;
    using size_type        = std::uint64_t;

    lockfree_fifo_backend(size_type initial_size = 0, size_type /* num_thread */ = size_type(-1)) : queue_(std::size_t(initial_size)) {}

    bool push(const_reference val, bool /*other_end*/ = false) { return queue_.enqueue(val); }

    bool push(rvalue_reference val, bool /*other_end*/ = false) { return queue_.enqueue(EINSUMS_MOVE(val)); }

    bool pop(reference val, bool /* steal */ = true) { return queue_.try_dequeue(val); }

    bool empty() { return (queue_.size_approx() == 0); }

  private:
    container_type queue_;
};

struct lockfree_fifo {
    template <typename T>
    struct apply {
        using type = lockfree_fifo_backend<T>;
    };
};

// LIFO
#if defined(EINSUMS_HAVE_CXX11_STD_ATOMIC_128BIT)
template <typename T>
struct lockfree_lifo_backend {
    using container_type =
        einsums::concurrency::detail::deque<T, einsums::concurrency::detail::caching_freelist_t, einsums::detail::aligned_allocator<T>>;

    using value_type       = T;
    using reference        = T &;
    using const_reference  = T const &;
    using rvalue_reference = T &&;
    using size_type        = std::uint64_t;

    lockfree_lifo_backend(size_type initial_size = 0, size_type /* num_thread */ = size_type(-1)) : queue_(std::size_t(initial_size)) {}

    bool push(const_reference val, bool other_end = false) {
        if (other_end)
            return queue_.push_right(val);
        return queue_.push_left(val);
    }

    bool push(rvalue_reference val, bool other_end = false) {
        if (other_end)
            return queue_.push_right(EINSUMS_MOVE(val));
        return queue_.push_left(EINSUMS_MOVE(val));
    }

    bool pop(reference val, bool /* steal */ = true) { return queue_.pop_left(val); }

    bool empty() { return queue_.empty(); }

  private:
    container_type queue_;
};

struct lockfree_lifo {
    template <typename T>
    struct apply {
        using type = lockfree_lifo_backend<T>;
    };
};

////////////////////////////////////////////////////////////////////////////
template <typename T>
struct lockfree_abp_fifo_backend {
    using container_type =
        einsums::concurrency::detail::deque<T, einsums::concurrency::detail::caching_freelist_t, einsums::detail::aligned_allocator<T>>;

    using value_type       = T;
    using reference        = T &;
    using const_reference  = T const &;
    using rvalue_reference = T &&;
    using size_type        = std::uint64_t;

    lockfree_abp_fifo_backend(size_type initial_size = 0, size_type /* num_thread */ = size_type(-1)) : queue_(std::size_t(initial_size)) {}

    bool push(const_reference val, bool /*other_end*/ = false) { return queue_.push_left(val); }

    bool push(rvalue_reference val, bool /*other_end*/ = false) { return queue_.push_left(EINSUMS_MOVE(val)); }

    bool pop(reference val, bool steal = true) {
        if (steal)
            return queue_.pop_left(val);
        return queue_.pop_right(val);
    }

    bool empty() { return queue_.empty(); }

  private:
    container_type queue_;
};

struct lockfree_abp_fifo {
    template <typename T>
    struct apply {
        using type = lockfree_abp_fifo_backend<T>;
    };
};

////////////////////////////////////////////////////////////////////////////
// LIFO + stealing at opposite end.
// E.g. ABP (Arora, Blumofe and Plaxton) queuing
// http://dl.acm.org/citation.cfm?id=277678
template <typename T>
struct lockfree_abp_lifo_backend {
    using container_type =
        einsums::concurrency::detail::deque<T, einsums::concurrency::detail::caching_freelist_t, einsums::detail::aligned_allocator<T>>;

    using value_type       = T;
    using reference        = T &;
    using const_reference  = T const &;
    using rvalue_reference = T &&;
    using size_type        = std::uint64_t;

    lockfree_abp_lifo_backend(size_type initial_size = 0, size_type /* num_thread */ = size_type(-1)) : queue_(std::size_t(initial_size)) {}

    bool push(const_reference val, bool other_end = false) {
        if (other_end)
            return queue_.push_right(EINSUMS_MOVE(val));
        return queue_.push_left(EINSUMS_MOVE(val));
    }

    bool push(rvalue_reference val, bool other_end = false) {
        if (other_end)
            return queue_.push_right(val);
        return queue_.push_left(val);
    }

    bool pop(reference val, bool steal = true) {
        if (steal)
            return queue_.pop_right(val);
        return queue_.pop_left(val);
    }

    bool empty() { return queue_.empty(); }

  private:
    container_type queue_;
};

struct lockfree_abp_lifo {
    template <typename T>
    struct apply {
        using type = lockfree_abp_lifo_backend<T>;
    };
};

#endif // EINSUMS_HAVE_CXX11_STD_ATOMIC_128BIT

} // namespace einsums::threads::detail
