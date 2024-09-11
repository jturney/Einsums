//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/modules/runtime_configuration.hpp>

namespace einsums::detail {

struct EINSUMS_EXPORT runtime;

/// The function \a get_runtime returns a reference to the (thread
/// specific) runtime instance.
EINSUMS_EXPORT runtime  &get_runtime();
EINSUMS_EXPORT runtime *&get_runtime_ptr();

EINSUMS_EXPORT einsums::util::runtime_configuration const &get_config();

EINSUMS_EXPORT void on_exit() noexcept;
EINSUMS_EXPORT void on_abort(int signal) noexcept;

} // namespace einsums::detail