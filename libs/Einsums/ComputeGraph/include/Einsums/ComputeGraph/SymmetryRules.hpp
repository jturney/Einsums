//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file SymmetryRules.hpp
 * @brief What permutational symmetry an operation's output has, from the operation and its inputs.
 *
 * @par Why this is its own component
 * More than one pass asks what a tensor's symmetry is. @ref passes::SymmetryPropagation asks it to
 * tag intermediates for the symmetric BLAS dispatch, and a pass that skips work a symmetry makes
 * redundant asks it to prove the skip is sound. Two derivations of one fact disagree in the corner
 * nobody tested, and here the disagreement is a wrong number: a tensor one pass believes symmetric
 * and the other does not. So the rules live once, here, and every pass that needs them calls them.
 *
 * @par What a rule promises
 * A rule answers for ONE node: given the descriptors its inputs carry, which symmetry its output is
 * guaranteed to have once that node has run. It says nothing about whether another node writes the
 * same tensor afterwards, or a child sub-graph does; that is the caller's soundness guard (see
 * @ref EscapeAnalysis::stable), because what counts as "nothing else can write it" depends on what
 * the caller is about to do with the answer.
 *
 * Every rule is unconditional. A rule that holds only for some values, or only up to a tolerance the
 * operation could exceed, is not a rule here. No answer is always safe: it means "nothing is known",
 * and the dispatch that would have used a symmetry falls back to the general path.
 */

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>

#include <optional>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;
struct Node;

/**
 * @brief The symmetry @p node's single output unconditionally has once the node has run.
 *
 * Reads only the node and the @c symmetry_hint of its inputs. The rules, one per op kind:
 *
 *  - **Scale**, @c C = alpha A: C has A's symmetry.
 *  - **Axpby**, @c C = alpha A + beta B: when A and B carry identical descriptors, C has that one.
 *  - **Einsum** with the same tensor in both operand slots, rank 2 by rank 2 into rank 2 over one
 *    summed index (A^T A and its relatives): C is symmetric in its two axes, or Hermitian when the
 *    element type is complex and exactly one operand is conjugated.
 *  - **Permute** or **Transpose** of a rank-2 input with a symmetric or Hermitian pair descriptor and
 *    a zero destination prefactor: C has the input's descriptor.
 *
 * @param[in] graph The graph @p node belongs to, for its inputs' handles.
 * @param[in] node  The node.
 * @return The descriptor, or nothing when no rule applies.
 */
[[nodiscard]] EINSUMS_EXPORT std::optional<SymmetryDescriptor> output_symmetry(Graph const &graph, Node const &node);

EINSUMS_NAMESPACE_END(compute_graph)
