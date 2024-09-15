//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/affinity/affinity_data.hpp>
#include <einsums/ini/ini.hpp>
#include <einsums/resource_partitioner/partitioner_fwd.hpp>

namespace einsums::resource::detail {
EINSUMS_EXPORT partitioner &create_partitioner(resource::partitioner_mode rpmode, einsums::detail::section rtcfg,
                                               einsums::detail::affinity_data affinity_data);

} // namespace einsums::resource::detail
