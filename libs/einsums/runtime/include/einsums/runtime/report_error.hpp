//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstddef>
#include <exception>

namespace einsums::detail {

/// Reports the given exception to the console
EINSUMS_EXPORT void report_error(std::size_t num_thread, std::exception_ptr const &e);

/// Reports the given exception to the console
EINSUMS_EXPORT void report_error(std::exception_ptr const &e);

} // namespace einsums::detail