//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/config.hpp>
#include <einsums/schedulers/deadlock_detection.hpp>

namespace einsums::threads::detail {
#ifdef EINSUMS_HAVE_THREAD_DEADLOCK_DETECTION
    static bool deadlock_detection_enabled = false;

    void set_deadlock_detection_enabled(bool enabled) { deadlock_detection_enabled = enabled; }

    bool get_deadlock_detection_enabled() { return deadlock_detection_enabled; }
#endif
}    // namespace einsums::threads::detail
