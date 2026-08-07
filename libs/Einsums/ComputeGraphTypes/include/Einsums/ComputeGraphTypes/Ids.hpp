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
 * Assigned sequentially by Graph::add_node(). Used internally for
 * adjacency tracking and debugging output.
 */
using NodeId = uint64_t;

/**
 * @brief Unique identifier for a tensor within a computation graph.
 *
 * Each tensor registered with a Graph receives a unique TensorId.
 * These IDs are used in Node::inputs and Node::outputs to express
 * data dependencies between operations.
 */
using TensorId = uint64_t;

EINSUMS_NAMESPACE_END(compute_graph)
