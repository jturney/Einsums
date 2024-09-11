//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <functional>

namespace einsums {

using shutdown_function_type = std::function<void()>;

EINSUMS_EXPORT void register_pre_shutdown_function(shutdown_function_type f);

EINSUMS_EXPORT void register_shutdown_function(shutdown_function_type f);

} // namespace einsums