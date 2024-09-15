//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

namespace einsums::detail {
// helper function for counter evaluation
inline std::uint64_t get_and_reset_value(std::uint64_t &value, bool reset) noexcept {
    std::uint64_t result = value;
    if (reset)
        value = 0;
    return result;
}

inline std::int64_t get_and_reset_value(std::int64_t &value, bool reset) noexcept {
    std::int64_t result = value;
    if (reset)
        value = 0;
    return result;
}

template <typename T>
inline T get_and_reset_value(std::atomic<T> &value, bool reset) noexcept {
    if (reset)
        return value.exchange(0, std::memory_order_acq_rel);
    return value.load(std::memory_order_relaxed);
}

inline std::vector<std::int64_t> get_and_reset_value(std::vector<std::int64_t> &value, bool reset) noexcept {
    std::vector<std::int64_t> result = value;
    if (reset)
        value.clear();

    return result;
}
} // namespace einsums::detail
