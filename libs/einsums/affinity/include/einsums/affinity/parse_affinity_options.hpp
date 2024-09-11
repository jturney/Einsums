//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/errors/error_code.hpp>
#include <einsums/topology/cpu_mask.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace einsums::detail {

enum distribution_type { compact = 0x01, scatter = 0x02, balanced = 0x04, numa_balanced = 0x08 };

using mappings_type = distribution_type;

EINSUMS_EXPORT void parse_mappings(std::string const &spec, mappings_type &mappings, error_code &ec = throws);

EINSUMS_EXPORT void parse_affinity_options(std::string const &spec, std::vector<threads::detail::mask_type> &affinities,
                                           std::size_t used_cores, std::size_t max_cores, std::size_t num_threads,
                                           std::vector<std::size_t> &num_pus, bool use_process_mask, error_code &ec = throws);

} // namespace einsums::detail