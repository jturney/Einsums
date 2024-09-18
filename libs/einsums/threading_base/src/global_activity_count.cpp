//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/threading_base/detail/global_activity_count.hpp>

#include <atomic>
#include <cstddef>

namespace einsums::threads::detail {
static std::atomic<std::size_t> global_activity_count{0};

void increment_global_activity_count() {
    global_activity_count.fetch_add(1, std::memory_order_acquire);
}

void decrement_global_activity_count() {
    global_activity_count.fetch_sub(1, std::memory_order_release);
}

std::size_t get_global_activity_count() {
    return global_activity_count.load(std::memory_order_acquire);
}
} // namespace einsums::threads::detail
