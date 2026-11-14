//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file SpaceTable.hpp
/// @brief What a graph has learned about its index spaces: which symbol each ties to, what
///        each measures, how each is cut into tiles, and the registry their ids came from.

#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {

/**
 * @brief A graph's index-space facts.
 *
 * Graph-level rather than per-handle, because the properties worth enforcing are global ones:
 * one symbol standing for two spaces, or one space measuring two extents, is a contradiction no
 * single handle can see. The table records; @ref Graph decides which disagreement is an error
 * and words it.
 *
 * An extent or a tiling is "uniform" while every statement about it agrees. Two that disagree
 * leave the entry known but unusable, which for an extent is a ragged family rather than a
 * mistake.
 */
class EINSUMS_EXPORT SpaceTable {
  public:
    /// The registry the ids on this graph's handles were issued by: the one set, or the
    /// process-global one.
    [[nodiscard]] SpaceRegistry &registry() const noexcept;

    /// Use @p registry for this graph's ids. Non-owning: a registry is a long-lived object the
    /// caller owns, and the graph stays movable.
    void set_registry(SpaceRegistry &registry) noexcept { _registry = &registry; }

    /// Symbol name to the space it ranges over.
    [[nodiscard]] std::unordered_map<std::string, SpaceId> const &symbol_spaces() const noexcept { return _symbol_spaces; }

    /// Tie @p symbol to @p space.
    /// @return The space @p symbol was already tied to when that differs, else nothing.
    [[nodiscard]] std::optional<SpaceId> tie_symbol(std::string const &symbol, SpaceId space);

    /// Record that an axis over @p space measures @p extent; a different one makes it ragged.
    void learn_extent(SpaceId space, std::size_t extent);

    /// The extent of @p space, when it is known and uniform.
    [[nodiscard]] std::optional<std::size_t> extent(SpaceId space) const noexcept;

    /// Whether @p space has been given extents that disagree.
    [[nodiscard]] bool extent_ragged(SpaceId space) const noexcept;

    /// State the extent of @p space outright; 0 forgets it.
    void pin_extent(SpaceId space, std::size_t extent);

    /// The canonical partition of @p space, when it is known and uniform.
    [[nodiscard]] std::optional<std::vector<int>> tiling(SpaceId space) const;

    /// Whether @p space has been given two different partitions.
    [[nodiscard]] bool tiling_conflicted(SpaceId space) const noexcept;

    /// Forget the partition of @p space.
    void clear_tiling(SpaceId space) { _tiles.erase(space); }

    /// State a partition of @p space whose tiles sum to @p total, and the extent that implies.
    /// A second, different partition leaves the space with none usable.
    void pin_tiling(SpaceId space, std::vector<int> const &tile_sizes, std::size_t total);

  private:
    std::unordered_map<std::string, SpaceId>                       _symbol_spaces;
    std::unordered_map<SpaceId, std::pair<std::size_t, bool>>      _extents; ///< extent, still uniform
    std::unordered_map<SpaceId, std::pair<std::vector<int>, bool>> _tiles;   ///< partition, still uniform
    SpaceRegistry                                                 *_registry{nullptr};
};

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
