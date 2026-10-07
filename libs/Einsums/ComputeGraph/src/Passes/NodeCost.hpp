//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file NodeCost.hpp
 * @brief The model price of one node, for the passes that plan by cost.
 *
 * Private to the passes. ThreadPlanning prices every node it has no measurement for from here,
 * and a registered descriptor's @c cost hook (DescriptorHooks.hpp) answers for its own node.
 */

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <functional>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

EINSUMS_NAMESPACE_END(compute_graph)

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// @brief What the model says a node costs at width 1 on a device, and how it widens.
struct ModelledCost {
    double       t1_us{0.0};                        ///< Serial time; 0 for a Loop or Conditional, priced by its body
    KernelFamily family{KernelFamily::Elementwise}; ///< The speedup curve it scales along
    std::size_t  bytes{0};                          ///< Working set, for the curve's size class
    /// A registered descriptor's own speedup curve, overriding @ref family's; empty otherwise.
    std::function<double(unsigned)> speedup;
};

/// @brief The model price of @p node in @p graph on @p profile, from its registered @c cost hook
///        when it has one that answers, otherwise from its kind and shape.
[[nodiscard]] ModelledCost model_cost(Graph const &graph, Node const &node, DeviceProfile const &profile);

EINSUMS_NAMESPACE_END(compute_graph::detail)
