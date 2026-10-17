//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file StorageCost.hpp
/// @brief What moving a box of a backed tensor costs: its runs of the storage, and the weights that
///        put a stored byte and a run on the scale of memory traffic.
///
/// Shared by SlabStreaming, which chooses the order a tensor is stored in from the boxes it moves,
/// and AxisTiling, which chooses the loop that decides those boxes, so the two count a box alike.

#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::detail)

/// The declared order of @p rank axes: ``0, 1, ..., rank - 1``.
[[nodiscard]] inline std::vector<std::size_t> declared_order(std::size_t rank) {
    std::vector<std::size_t> out(rank);
    for (std::size_t p = 0; p < rank; ++p) {
        out[p] = p;
    }
    return out;
}

/// The contiguous runs of a column-major storage a box of @p extents costs when its axes are stored
/// in @p order: every leading stored axis the box covers whole is one run with the next.
/// @param[in] order The declared axis at each stored position.
/// @param[in] extents The box's extent on each declared axis.
/// @param[in] whole_axis Whether the box covers each declared axis whole.
[[nodiscard]] inline std::size_t runs_in(std::vector<std::size_t> const &order, std::vector<std::size_t> const &extents,
                                         std::vector<bool> const &whole_axis) {
    std::size_t inner = 0;
    while (inner + 1 < order.size() && whole_axis[order[inner]]) {
        ++inner;
    }
    std::size_t runs = 1;
    for (std::size_t s = inner + 1; s < order.size(); ++s) {
        runs *= extents[order[s]];
    }
    return runs;
}

/// The fewest runs any stored order gives the box: the axes it covers whole stored first, then the
/// partial ones with the longest of them first, so it is the run and the others multiply.
[[nodiscard]] inline std::size_t fewest_runs(std::vector<std::size_t> const &extents, std::vector<bool> const &whole_axis) {
    std::size_t product = 1;
    std::size_t longest = 1;
    for (std::size_t p = 0; p < extents.size(); ++p) {
        if (!whole_axis[p]) {
            product *= extents[p];
            longest = std::max(longest, extents[p]);
        }
    }
    return product / longest;
}

/// The bytes of memory traffic one byte moved to or from storage counts as.
[[nodiscard]] inline long double storage_byte_weight() {
    return static_cast<long double>(config::get(option::GraphStorageByteWeight));
}

/// The bytes of memory traffic one contiguous run of storage costs beyond its bytes.
[[nodiscard]] inline long double storage_run_bytes() {
    return static_cast<long double>(config::get(option::GraphStorageRunBytes));
}

EINSUMS_NAMESPACE_END(compute_graph::passes::detail)
