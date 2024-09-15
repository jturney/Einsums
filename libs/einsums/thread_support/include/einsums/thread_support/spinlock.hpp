//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <atomic>

namespace einsums::detail {

/// Lockable spinlock class
struct spinlock {
  public:
    EINSUMS_NON_COPYABLE(spinlock);

  private:
    std::atomic<bool> m;

    EINSUMS_EXPORT void yield_k(unsigned) noexcept;

  public:
    constexpr spinlock() noexcept : m(false) {}

    EINSUMS_FORCEINLINE bool try_lock() noexcept {
        // First do a relaxed load to check if lock is free in order to prevent
        // unnecessary cache misses if someone does while(!try_lock())
        return !m.load(std::memory_order_relaxed) && !m.exchange(true, std::memory_order_acquire);
    }

    void lock() noexcept {
        // Wait for lock to be released without generating cache misses
        // Similar implementation to einsums::concurrency::detail::spinlock
        unsigned k = 0;
        while (!try_lock()) {
            yield_k(k++);
        }
    }

    EINSUMS_FORCEINLINE void unlock() noexcept { m.store(false, std::memory_order_release); }
};

} // namespace einsums::detail
