//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>

#include <cstdint>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief Unique identifier for a node within a computation graph.
 *
 * Issued sequentially by the graph that holds the node. Keys update_prefactors,
 * timing samples, the dot and JSON exports and the pass program-order validator.
 */
using NodeId = uint64_t;

/**
 * @brief The id a node carries until a graph issues it one.
 *
 * A pass that builds a node, or moves one in from another graph, leaves or sets this
 * value, and the graph issues a fresh id when the pass hands it back. A node never keeps
 * an id issued by a different graph: two graphs number independently, so such an id would
 * name another node here.
 */
inline constexpr NodeId unassigned_node_id = static_cast<NodeId>(-1);

/**
 * @brief Unique identifier for a tensor within a computation graph.
 *
 * Each tensor registered with a Graph receives a unique TensorId.
 * These IDs are used in Node::inputs and Node::outputs to express
 * data dependencies between operations.
 */
using TensorId = uint64_t;

EINSUMS_NAMESPACE_END(compute_graph)
