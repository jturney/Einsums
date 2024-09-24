//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/config.hpp>
#include <einsums/schedulers/maintain_queue_wait_times.hpp>

namespace einsums::threads::detail {
#ifdef EINSUMS_HAVE_THREAD_QUEUE_WAITTIME
    static bool maintain_queue_wait_times_enabled = false;

    void set_maintain_queue_wait_times_enabled(bool enabled)
    {
        maintain_queue_wait_times_enabled = enabled;
    }

    bool get_maintain_queue_wait_times_enabled() { return maintain_queue_wait_times_enabled; }
#endif
}    // namespace einsums::threads::detail
