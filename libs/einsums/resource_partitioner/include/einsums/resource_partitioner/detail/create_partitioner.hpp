//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/ini/ini.hpp>
#include <einsums/resource_partitioner/partitioner_fwd.hpp>

namespace pika::resource::detail {
EINSUMS_EXPORT partitioner& create_partitioner(resource::partitioner_mode rpmode,
    pika::detail::section rtcfg, pika::detail::affinity_data affinity_data);

}    // namespace pika::resource::detail
