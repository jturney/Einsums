//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Comm/ProcessGrid.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace einsums::comm {

/// Which axis of the 2D process grid a tensor dimension is distributed along.
enum class GridAxis : std::uint8_t {
    None, ///< This dimension is replicated (not distributed)
    Row,  ///< Distributed across grid rows (Pr processes)
    Col,  ///< Distributed across grid columns (Pc processes)
};

/**
 * @brief Type-erased distribution descriptor for a tensor on a 2D process grid.
 *
 * Stored in TensorHandle::distribution_info as shared_ptr<DistributionDescriptor>.
 * Replaces the previous shared_ptr<size_t> (single dimension index).
 *
 * For a GEMM C[i,j] = A[i,k] * B[k,j] on a 2×2 grid:
 * - C: dim_to_axis = {Row, Col}, global_dims = {M, N}
 * - A: dim_to_axis = {Row, None}, global_dims = {M, K}
 * - B: dim_to_axis = {None, Col}, global_dims = {K, N}
 *
 * Each rank holds local partitions determined by its grid coordinates.
 */
struct DistributionDescriptor {
    std::vector<GridAxis> dim_to_axis;   ///< Per tensor dimension: which grid axis
    std::vector<size_t>   global_dims;   ///< Original (global) sizes before partitioning
    ProcessGrid const    *grid{nullptr}; ///< Process grid (not owned)
    bool                  summa{false};  ///< True if this tensor participates in a SUMMA contraction

    /// Check if this tensor is fully replicated (no distributed dimensions).
    [[nodiscard]] bool is_fully_replicated() const {
        return std::ranges::all_of(dim_to_axis, [](GridAxis a) { return a == GridAxis::None; });
    }

    /// Compute the local range [start, end) for a given rank along dimension @p dim.
    /// Uses balanced blocking: distributes remainder evenly across first processes.
    /// For N=41, P=4: {11, 11, 10, 9} instead of {11, 11, 11, 8}.
    [[nodiscard]] std::pair<size_t, size_t> local_range(size_t dim, int rank) const {
        if (dim >= dim_to_axis.size() || dim_to_axis[dim] == GridAxis::None)
            return {0, global_dims[dim]};

        auto [row, col] = grid->coords(rank);
        int nprocs;
        int my_pos;

        if (dim_to_axis[dim] == GridAxis::Row) {
            nprocs = grid->rows();
            my_pos = row;
        } else {
            nprocs = grid->cols();
            my_pos = col;
        }

        size_t const N         = global_dims[dim];
        auto const   P         = static_cast<size_t>(nprocs);
        size_t const base      = N / P; // Minimum elements per process
        size_t const remainder = N % P; // Extra elements to distribute

        // First 'remainder' processes get (base+1) elements, rest get 'base'.
        size_t start, count;
        if (std::cmp_less(my_pos, remainder)) {
            count = base + 1;
            start = static_cast<size_t>(my_pos) * (base + 1);
        } else {
            count = base;
            start = remainder * (base + 1) + (static_cast<size_t>(my_pos) - remainder) * base;
        }

        return {start, start + count};
    }

    /// Compute local dimensions for a given rank.
    [[nodiscard]] std::vector<size_t> local_dims_for(int rank) const {
        std::vector<size_t> result(dim_to_axis.size());
        for (size_t d = 0; d < dim_to_axis.size(); d++) {
            auto [start, end] = local_range(d, rank);
            result[d]         = end - start;
        }
        return result;
    }

    /// Check if a specific dimension is distributed (Row or Col).
    [[nodiscard]] bool is_dim_distributed(size_t dim) const { return dim < dim_to_axis.size() && dim_to_axis[dim] != GridAxis::None; }
};

} // namespace einsums::comm
