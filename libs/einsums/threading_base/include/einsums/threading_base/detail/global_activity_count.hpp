//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstddef>

namespace einsums::threads::detail {
EINSUMS_EXPORT void increment_global_activity_count();
EINSUMS_EXPORT void decrement_global_activity_count();
EINSUMS_EXPORT std::size_t get_global_activity_count();
} // namespace einsums::threads::detail
