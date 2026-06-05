//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file DistributionMap.hpp
/// @brief Tensor distribution strategies for MPI partitioning.
///
/// Used internally by ComputeGraph passes (DistributionPlanning, Materialization)
/// to decide how deferred tensors are partitioned across ranks. Users never interact
/// with these types directly — the pass infrastructure handles everything transparently.

#include <Einsums/Comm/Runtime.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace einsums::comm {

/// How a single dimension of a tensor is distributed across ranks.
enum class DistStrategy : std::uint8_t {
    Replicated, ///< Full copy on every rank
    Blocked,    ///< Contiguous block per rank (dim / nranks elements each)
    BlockCyclic ///< Round-robin blocks of fixed size
};

/**
 * @brief Describes how a tensor is partitioned across MPI ranks.
 *
 * Each dimension has an independent distribution strategy. The default
 * is `auto_distribute()` which replicates small tensors and block-distributes
 * large ones along the largest dimension.
 *
 * @tparam Rank Number of tensor dimensions.
 */
template <size_t Rank>
struct DistributionMap {
    std::array<DistStrategy, Rank> strategy{};
    std::array<size_t, Rank>       block_size{};
    std::array<size_t, Rank>       global_dims{};
    int                            num_ranks{1};

    /// Check if this distribution is fully replicated (all dims replicated).
    [[nodiscard]] constexpr bool is_replicated() const {
        return std::ranges::all_of(strategy, [](DistStrategy s) { return s == DistStrategy::Replicated; });
    }

    /// Number of blocks along dimension @p dim for this strategy.
    [[nodiscard]] size_t num_blocks(size_t dim) const {
        if (strategy[dim] == DistStrategy::Replicated)
            return 1;
        return (global_dims[dim] + block_size[dim] - 1) / block_size[dim];
    }

    /// Compute the range [start, end) of elements owned by @p rank along @p dim.
    [[nodiscard]] std::pair<size_t, size_t> local_range(size_t dim, int rank) const {
        if (strategy[dim] == DistStrategy::Replicated)
            return {0, global_dims[dim]};

        if (strategy[dim] == DistStrategy::Blocked) {
            size_t chunk = (global_dims[dim] + static_cast<size_t>(num_ranks) - 1) / static_cast<size_t>(num_ranks);
            size_t start = std::min(static_cast<size_t>(rank) * chunk, global_dims[dim]);
            size_t end   = std::min(start + chunk, global_dims[dim]);
            return {start, end};
        }

        // BlockCyclic: rank owns blocks at positions rank, rank+nranks, rank+2*nranks, ...
        auto   block_idx = static_cast<size_t>(rank);
        size_t start     = block_idx * block_size[dim];
        size_t end       = std::min(start + block_size[dim], global_dims[dim]);
        if (start >= global_dims[dim])
            return {0, 0};
        return {start, end};
    }

    /// Compute local dimensions for @p rank.
    [[nodiscard]] std::array<size_t, Rank> local_dims(int rank) const {
        std::array<size_t, Rank> dims{};
        for (size_t d = 0; d < Rank; d++) {
            auto [start, end] = local_range(d, rank);
            dims[d]           = end - start;
        }
        return dims;
    }
};

/**
 * @brief Auto-decide distribution: replicate if small, block-distribute if large.
 *
 * @param dims           Global tensor dimensions.
 * @param element_size   Size of one element in bytes (e.g., sizeof(double) = 8).
 * @param threshold      Tensors smaller than this (in bytes) are replicated.
 * @return A DistributionMap with the chosen strategy.
 */
template <size_t Rank>
[[nodiscard]] DistributionMap<Rank> auto_distribute(std::array<size_t, Rank> const &dims, size_t element_size = sizeof(double),
                                                    size_t threshold = static_cast<long>(64 * 1024) * 1024) {
    DistributionMap<Rank> map;
    map.global_dims = dims;
    map.num_ranks   = world_size();
    map.block_size  = dims;

    // Total bytes
    size_t total = element_size;
    for (auto d : dims)
        total *= d;

    if (total <= threshold || map.num_ranks <= 1) {
        map.strategy.fill(DistStrategy::Replicated);
        return map;
    }

    // Block-distribute along the largest dimension
    map.strategy.fill(DistStrategy::Replicated);

    size_t max_dim = 0;
    size_t max_idx = 0;
    for (size_t d = 0; d < Rank; d++) {
        if (dims[d] > max_dim) {
            max_dim = dims[d];
            max_idx = d;
        }
    }

    map.strategy[max_idx]   = DistStrategy::Blocked;
    map.block_size[max_idx] = (dims[max_idx] + static_cast<size_t>(map.num_ranks) - 1) / static_cast<size_t>(map.num_ranks);

    return map;
}

} // namespace einsums::comm
