//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <string>
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

    /// A registered handle to write, for a test that builds a state the graph's methods refuse.
    static TensorHandle &tensor(Graph &graph, TensorId id) { return graph.mutable_tensor(id); }

    /// Place a graph in a pipeline without building one, for a test of what a graph carries.
    static void set_stage(Graph &graph, std::string const &pipeline, std::string const &workspace, std::string const &stage,
                          std::string const &type, int index) {
        graph.set_stage(pipeline, workspace, stage, type, index);
    }
};

EINSUMS_NAMESPACE_END(compute_graph::detail)
