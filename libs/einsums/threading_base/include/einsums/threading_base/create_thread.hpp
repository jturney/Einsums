//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/modules/errors.hpp>
#include <einsums/threading_base/thread_init_data.hpp>
#include <einsums/threading_base/threading_base_fwd.hpp>

namespace einsums::threads::detail {
EINSUMS_EXPORT void create_thread(scheduler_base *scheduler, thread_init_data &data, thread_id_ref_type &id, error_code &ec = throws);
}