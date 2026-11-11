//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// The node list, written around the graph's API.
///
/// For a test that builds what no pass may, a stray node, an undeclared reorder, operand lists
/// that disagree with their tensors, so as to check that the guard meant to catch it does; and
/// for one that works beneath the graph's commit on purpose, installing what the executor builder
/// returns or writing a live parameter with no rebuild, to check the builder and the executors
/// alone. Everything else edits through Graph::edit and the node-set functions, as a pass does.
struct GraphBackdoor {
    static std::vector<Node> &nodes(Graph &graph) { return graph._nodes; }
};

EINSUMS_NAMESPACE_END(compute_graph::detail)
