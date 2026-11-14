//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file SpaceTable.cpp
/// @brief A graph's index-space facts, and the rule that two disagreeing statements leave a
///        fact known but unusable.

#include <Einsums/ComputeGraph/Detail/SpaceTable.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {

SpaceRegistry &SpaceTable::registry() const noexcept {
    return _registry != nullptr ? *_registry : global_space_registry();
}

std::optional<SpaceId> SpaceTable::tie_symbol(std::string const &symbol, SpaceId space) {
    auto const [it, inserted] = _symbol_spaces.emplace(symbol, space);
    if (!inserted && it->second != space) {
        return it->second;
    }
    return std::nullopt;
}

void SpaceTable::learn_extent(SpaceId space, std::size_t extent) {
    auto const [it, inserted] = _extents.try_emplace(space, extent, true);
    if (!inserted && it->second.first != extent) {
        // Not an error. Two axes over one space with different extents is exactly what a
        // ragged family is, and the design says so: PNO domains have a different virtual
        // extent per pair. What it costs is the ability to SIZE an axis from the space,
        // so the space stops being usable that way and says so if anyone asks.
        it->second.second = false;
    }
}

std::optional<std::size_t> SpaceTable::extent(SpaceId space) const noexcept {
    auto const it = _extents.find(space);
    if (it == _extents.end() || !it->second.second) {
        return std::nullopt;
    }
    return it->second.first;
}

bool SpaceTable::extent_ragged(SpaceId space) const noexcept {
    auto const it = _extents.find(space);
    return it != _extents.end() && !it->second.second;
}

void SpaceTable::pin_extent(SpaceId space, std::size_t extent) {
    if (extent == 0) {
        _extents.erase(space);
        return;
    }
    _extents[space] = {extent, true};
}

std::optional<std::vector<int>> SpaceTable::tiling(SpaceId space) const {
    auto const it = _tiles.find(space);
    if (it == _tiles.end() || !it->second.second) {
        return std::nullopt;
    }
    return it->second.first;
}

bool SpaceTable::tiling_conflicted(SpaceId space) const noexcept {
    auto const it = _tiles.find(space);
    return it != _tiles.end() && !it->second.second;
}

void SpaceTable::pin_tiling(SpaceId space, std::vector<int> const &tile_sizes, std::size_t total) {
    auto const [it, inserted] = _tiles.try_emplace(space, tile_sizes, true);
    if (!inserted && it->second.first != tile_sizes) {
        // Mirrors the extent rule: a second, different statement is not an error but it does
        // leave the space without a canonical answer, so an axis has to bring its own.
        it->second.second = false;
    }
    _extents[space] = {total, true};
}

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
