//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

namespace einsums::threads::detail {
#ifdef EINSUMS_HAVE_THREAD_QUEUE_WAITTIME
EINSUMS_EXPORT void set_maintain_queue_wait_times_enabled(bool enabled);
EINSUMS_EXPORT bool get_maintain_queue_wait_times_enabled();
#endif
} // namespace einsums::threads::detail
