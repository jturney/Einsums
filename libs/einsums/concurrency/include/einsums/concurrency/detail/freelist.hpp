//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <boost/lockfree/policies.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/version.hpp>
#include <cstddef>

namespace einsums::concurrency::detail {
template <typename T, typename Alloc = std::allocator<T>>
class caching_freelist : public boost::lockfree::detail::freelist_stack<T, Alloc> {
    using base_type = boost::lockfree::detail::freelist_stack<T, Alloc>;

  public:
    caching_freelist(std::size_t n = 0) : boost::lockfree::detail::freelist_stack<T, Alloc>(Alloc(), n) {}

    T *allocate() { return this->base_type::template allocate<true, false>(); }

    void deallocate(T *n) { this->base_type::template deallocate<true>(n); }
};

template <typename T, typename Alloc = std::allocator<T>>
class static_freelist : public boost::lockfree::detail::freelist_stack<T, Alloc> {
    using base_type = boost::lockfree::detail::freelist_stack<T, Alloc>;

  public:
    static_freelist(std::size_t n = 0) : boost::lockfree::detail::freelist_stack<T, Alloc>(Alloc(), n) {}

    T *allocate() { return this->base_type::template allocate<true, true>(); }

    void deallocate(T *n) { this->base_type::template deallocate<true>(n); }
};

struct caching_freelist_t {};
struct static_freelist_t {};
} // namespace einsums::concurrency::detail
