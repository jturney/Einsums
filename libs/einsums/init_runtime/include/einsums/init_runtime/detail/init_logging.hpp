//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/runtime_configuration/runtime_configuration.hpp>

namespace einsums::detail {
EINSUMS_EXPORT void init_logging(util::runtime_configuration &ini);
}