//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <atomic>

namespace einsums::detail {

struct atomic_count {
    EINSUMS_NON_COPYABLE(atomic_count);

    explicit atomic_count(long value) : _value(value) {}

    auto operator=(long value) -> atomic_count & {
        _value.store(value, std::memory_order_relaxed);
        return *this;
    }

    auto operator++() -> long { return _value.fetch_add(1, std::memory_order_acq_rel) + 1; }

    auto operator--() -> long { return _value.fetch_sub(1, std::memory_order_acq_rel) - 1; }

    auto operator+=(long n) -> atomic_count & {
        _value.fetch_add(n, std::memory_order_acq_rel);
        return *this;
    }

    auto operator-=(long n) -> atomic_count & {
        _value.fetch_sub(n, std::memory_order_acq_rel);
        return *this;
    }

    operator long() const { return _value.load(std::memory_order_acquire); }

  private:
    std::atomic<long> _value;
};

} // namespace einsums::detail