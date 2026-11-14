//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file NodeList.cpp
/// @brief Issuing node ids, and the two analyses a graph caches against its node list.

#include <Einsums/ComputeGraph/Detail/NodeList.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {

NodeId NodeList::append(Node node) {
    node.id         = _next_node_id++;
    NodeId const id = node.id;
    _nodes.push_back(std::move(node));
    return id;
}

void NodeList::assign_ids() noexcept {
    for (auto &node : _nodes) {
        if (node.id == unassigned_node_id) {
            node.id = _next_node_id++;
        }
    }
}

std::span<std::uint32_t const> NodeList::subgraph_positions() {
    if (_subgraph_positions_version != _analysis_version || _subgraph_positions_count != _nodes.size()) {
        _subgraph_positions.clear();
        for (std::size_t i = 0; i < _nodes.size(); ++i) {
            bool holds = false;
            for_each_child_graph(_nodes[i], [&holds](Graph &) { holds = true; });
            if (holds) {
                _subgraph_positions.push_back(static_cast<std::uint32_t>(i));
            }
        }
        _subgraph_positions_version = _analysis_version;
        _subgraph_positions_count   = _nodes.size();
    }
    return _subgraph_positions;
}

UsageAnalysis const &NodeList::usage(Graph &graph, std::uint64_t subtree_stamp) {
    if (_usage_version != _analysis_version || _usage.node_count() != _nodes.size() || _usage_subtree_stamp != subtree_stamp) {
        _usage               = UsageAnalysis::build(graph);
        _usage_version       = _analysis_version;
        _usage_subtree_stamp = subtree_stamp;
    }
    return _usage;
}

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
