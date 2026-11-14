//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file NodeList.hpp
/// @brief A graph's nodes in program order, the dependency lists built over them, the counters
///        that say when the graph last changed, and the two analyses cached against them.

#include <Einsums/ComputeGraph/Executor.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/UsageAnalysis.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

namespace detail {

/**
 * @brief The nodes of one graph and what is derived from their order.
 *
 * The vector is the program: a node's position is what the hazard scan orders it by, so the
 * dependency lists are keyed by position and go stale with any edit. Two flags say how stale.
 * The order is "sorted" once a sort produced it or a caller vouched for it; the lists are
 * current only while the order is, the node count matches them, and no body below the graph
 * changed since they were built (the sub-graph stamp, which the graph computes).
 *
 * The analysis version moves at every change a per-buffer analysis can see, and the structure
 * version at every change to the node set; ``Graph::note`` decides which change moves which.
 */
class EINSUMS_EXPORT NodeList {
  public:
    [[nodiscard]] std::vector<Node>       &nodes() noexcept { return _nodes; }
    [[nodiscard]] std::vector<Node> const &nodes() const noexcept { return _nodes; }

    /// A fresh id, for a node built before it is added.
    NodeId reserve_id() noexcept { return _next_node_id++; }

    /// Append @p node with a fresh id; the id it had is discarded.
    /// @return The id given.
    NodeId append(Node node);

    /// Give every node still holding ``unassigned_node_id`` a fresh id.
    void assign_ids() noexcept;

    /// Whether the order is known topological: a sort produced it or a caller vouched for it.
    [[nodiscard]] bool sorted() const noexcept { return _sorted; }

    /// The caller vouches for the order; the lists stay as stale as they were.
    void vouch_order() noexcept { _sorted = true; }

    /// The order is no longer known to be topological.
    void invalidate_order() noexcept { _sorted = false; }

    [[nodiscard]] DependencyInfo       &deps() noexcept { return _deps; }
    [[nodiscard]] DependencyInfo const &deps() const noexcept { return _deps; }

    /// Whether the order is sorted and the lists were built for it, for this many nodes, at
    /// @p subtree_stamp.
    [[nodiscard]] bool deps_current(std::uint64_t subtree_stamp) const noexcept {
        return _sorted && _deps_valid && _deps.successors.size() == _nodes.size() && _deps_subtree_stamp == subtree_stamp;
    }

    /// The lists no longer match the nodes.
    void invalidate_deps() noexcept { _deps_valid = false; }

    /// The order is sorted and the lists were just built for it at @p subtree_stamp.
    void deps_built(std::uint64_t subtree_stamp) noexcept {
        _sorted             = true;
        _deps_valid         = true;
        _deps_subtree_stamp = subtree_stamp;
    }

    [[nodiscard]] std::uint64_t analysis_version() const noexcept { return _analysis_version; }
    void                        bump_analysis_version() noexcept { ++_analysis_version; }

    [[nodiscard]] std::uint64_t structure_version() const noexcept { return _structure_version; }
    void                        bump_structure_version() noexcept { ++_structure_version; }

    /// Positions of the nodes that hold a sub-graph, found again when the analysis version or
    /// the node count moved since the last call.
    std::span<std::uint32_t const> subgraph_positions();

    /// The usage table of @p graph (whose list this is), rebuilt when the analysis version, the
    /// node count or @p subtree_stamp moved since it was built.
    UsageAnalysis const &usage(Graph &graph, std::uint64_t subtree_stamp);

  private:
    std::vector<Node> _nodes;
    NodeId            _next_node_id{0};

    bool _sorted{false};

    /// Whether _deps matches the current node order. Distinct from _sorted: vouching for the
    /// order does not rebuild the lists, so a sort can skip the Kahn pass but must refresh them.
    bool _deps_valid{false};

    /// Sub-graph stamp _deps was built at: a control-flow node's edges come from its bodies, so a
    /// declared edit below this graph makes them stale too.
    std::uint64_t _deps_subtree_stamp{0};

    /// Bumped at every change a per-buffer analysis can see; usage() is keyed on it.
    std::uint64_t _analysis_version{0};
    /// Bumped at every change to the node set, for the pass phase rule.
    std::uint64_t _structure_version{0};

    /// Version _usage was built at (UINT64_MAX = never built).
    std::uint64_t _usage_version{std::numeric_limits<std::uint64_t>::max()};
    UsageAnalysis _usage;
    /// Sub-graph stamp _usage was built at.
    std::uint64_t _usage_subtree_stamp{0};

    /// Positions rather than pointers: a body is re-read from the node each time, so a body
    /// swapped in a node's descriptor is never held past its life.
    std::vector<std::uint32_t> _subgraph_positions;
    std::uint64_t              _subgraph_positions_version{std::numeric_limits<std::uint64_t>::max()};
    std::size_t                _subgraph_positions_count{0};

    DependencyInfo _deps;
};

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
