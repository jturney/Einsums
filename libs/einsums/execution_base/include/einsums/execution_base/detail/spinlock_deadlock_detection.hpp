//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstddef>

#ifdef EINSUMS_HAVE_SPINLOCK_DEADLOCK_DETECTION
namespace einsums::util::detail {
EINSUMS_EXPORT void set_spinlock_break_on_deadlock_enabled(bool enabled);
EINSUMS_EXPORT bool get_spinlock_break_on_deadlock_enabled();
EINSUMS_EXPORT void set_spinlock_deadlock_detection_limit(std::size_t limit);
EINSUMS_EXPORT void set_spinlock_deadlock_warning_limit(std::size_t limit);
EINSUMS_EXPORT std::size_t get_spinlock_deadlock_detection_limit();
EINSUMS_EXPORT std::size_t get_spinlock_deadlock_warning_limit();
} // namespace einsums::util::detail
#endif
