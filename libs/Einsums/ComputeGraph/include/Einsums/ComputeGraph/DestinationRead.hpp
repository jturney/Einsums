//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file DestinationRead.hpp
/// @brief Whether a node reads the tensor it writes, and the one way its input list says so.
///
/// A node whose destination carries a prefactor (an einsum's C prefactor, an axpby's or a
/// permute's beta, a product's beta) reads its destination exactly when that prefactor is not
/// zero. Two parts of the library need the answer. The passes that rewrite ask the descriptor.
/// The schedulers and the liveness passes see a read only through the node's input list. So the
/// input list follows one rule: the kind's operands first, then the destination once, as a
/// trailing entry, exactly when the node reads it.
///
/// The rule is applied where nodes are made (capture and Graph::add_node), after every pass (a
/// pass may rewrite a prefactor in place), by Graph::update_prefactors and by the GraphIR loader,
/// and Graph::verify reports a node that breaks it.

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <optional>
#include <span>
#include <string>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

/**
 * @brief True when @p node reads its destination (accumulates into it or updates it in place).
 *
 * The in-place scale and element transform always do. An axpby without a descriptor is assumed to,
 * since nothing says otherwise. Every other kind with a destination prefactor reads its
 * destination exactly when the LIVE prefactor is nonzero, so a pass that rewrites the prefactor is
 * answered by its rewrite. A node without a destination prefactor (a dot, a trace, a grouped node)
 * returns false.
 */
[[nodiscard]] EINSUMS_EXPORT bool reads_destination(Node const &node);

/**
 * @brief How many operands @p node lists before its destination, when the destination rule covers it.
 *
 * Empty for a node the rule does not cover: no destination prefactor, more than one output (a
 * grouped or batched node keeps one prefactor per member), or a descriptor-less Custom node.
 */
[[nodiscard]] EINSUMS_EXPORT std::optional<std::size_t> destination_operand_count(Node const &node);

/**
 * @brief The inputs @p node reads besides its destination: its input list without the trailing
 *        destination entry. All of the inputs for a node the rule does not cover.
 */
[[nodiscard]] EINSUMS_EXPORT std::span<TensorId const> operand_inputs(Node const &node);

/**
 * @brief Make @p node's input list follow the rule: its operands, then its destination exactly
 *        when it reads it. A node the rule does not cover is left alone.
 */
EINSUMS_EXPORT void sync_destination_input(Node &node);

/**
 * @brief Why @p node breaks the destination rule, or empty when it keeps it.
 *
 * Compared by buffer through @p graph, so a destination listed through a view of it counts.
 */
[[nodiscard]] EINSUMS_EXPORT std::optional<std::string> destination_rule_violation(Graph const &graph, Node const &node);

EINSUMS_NAMESPACE_END(compute_graph)
