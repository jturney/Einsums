//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Hardware/CPUMask.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace einsums::hardware {

enum class Mapping { Compact = 0x01, Scatter = 0x02, Balanced = 0x04, NumaBalanced = 0x08 };

EINSUMS_EXPORT void parse_mappings(std::string const &spec, Mapping &mappings);

EINSUMS_EXPORT void parse_affinity_options(std::string const &spec, std::vector<MaskType> &affinities, std::size_t used_cores,
                                           std::size_t max_cores, std::size_t num_threads, std::vector<std::size_t> &num_pus,
                                           bool use_process_mask);

} // namespace einsums::hardware