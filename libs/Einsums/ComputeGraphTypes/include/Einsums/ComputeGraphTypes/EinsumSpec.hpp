//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

namespace einsums::compute_graph {

/**
 * @brief Result of parsing an einsum specification string.
 *
 * Contains the parsed index lists for the output tensor (C) and the two
 * input tensors (A and B). Also stores the original string for error messages.
 */
struct ParsedEinsumSpec {
    std::vector<std::string> c_indices; ///< Output (C) indices
    std::vector<std::string> a_indices; ///< First input (A) indices
    std::vector<std::string> b_indices; ///< Second input (B) indices
    std::string              raw;       ///< Original specification string
};

} // namespace einsums::compute_graph
