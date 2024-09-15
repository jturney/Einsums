//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/execution_base/detail/spinlock_deadlock_detection.hpp>

#include <cstddef>

#ifdef EINSUMS_HAVE_SPINLOCK_DEADLOCK_DETECTION
namespace einsums::util::detail {
static bool        spinlock_break_on_deadlock_enabled = false;
static std::size_t spinlock_deadlock_detection_limit  = EINSUMS_SPINLOCK_DEADLOCK_DETECTION_LIMIT;
static std::size_t spinlock_deadlock_warning_limit    = EINSUMS_SPINLOCK_DEADLOCK_WARNING_LIMIT;

void set_spinlock_break_on_deadlock_enabled(bool enabled) {
    spinlock_break_on_deadlock_enabled = enabled;
}

bool get_spinlock_break_on_deadlock_enabled() {
    return spinlock_break_on_deadlock_enabled;
}

void set_spinlock_deadlock_detection_limit(std::size_t limit) {
    spinlock_deadlock_detection_limit = limit;
}

void set_spinlock_deadlock_warning_limit(std::size_t limit) {
    spinlock_deadlock_warning_limit = limit;
}

std::size_t get_spinlock_deadlock_detection_limit() {
    return spinlock_deadlock_detection_limit;
}

std::size_t get_spinlock_deadlock_warning_limit() {
    return spinlock_deadlock_warning_limit;
}
} // namespace einsums::util::detail
#endif
