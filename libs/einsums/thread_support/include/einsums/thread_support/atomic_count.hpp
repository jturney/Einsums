//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <atomic>

namespace einsums::detail {
class atomic_count {
  public:
    EINSUMS_NON_COPYABLE(atomic_count);

  public:
    explicit atomic_count(long value) : value_(value) {}

    atomic_count &operator=(long value) {
        value_.store(value, std::memory_order_relaxed);
        return *this;
    }

    long operator++() { return value_.fetch_add(1, std::memory_order_acq_rel) + 1; }

    long operator--() { return value_.fetch_sub(1, std::memory_order_acq_rel) - 1; }

    atomic_count &operator+=(long n) {
        value_.fetch_add(n, std::memory_order_acq_rel);
        return *this;
    }

    atomic_count &operator-=(long n) {
        value_.fetch_sub(n, std::memory_order_acq_rel);
        return *this;
    }

    operator long() const { return value_.load(std::memory_order_acquire); }

  private:
    std::atomic<long> value_;
};
} // namespace einsums::detail
