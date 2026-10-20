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

#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>

#include <optional>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class EscapeAnalysis;
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
 *  - **Einsum** of any rank with the same tensor in both operand slots and a zero destination
 *    prefactor: C has the symmetry @ref self_product_symmetry finds.
 *  - **Permute** or **Transpose** of a rank-2 input with a symmetric or Hermitian pair descriptor and
 *    a zero destination prefactor: C has the input's descriptor.
 *
 * @param[in] graph The graph @p node belongs to, for its inputs' handles.
 * @param[in] node  The node.
 * @return The descriptor, or nothing when no rule applies.
 */
[[nodiscard]] EINSUMS_EXPORT std::optional<SymmetryDescriptor> output_symmetry(Graph const &graph, Node const &node);

/**
 * @brief The symmetry the product term of an einsum has when one tensor fills both operand slots.
 *
 * Swapping the two operands of @c A[a...] @c A[b...] changes nothing, so when a relabeling of the
 * letters carries A's list onto B's and B's onto A's, the term is unchanged by that relabeling. If
 * it maps C's letters onto C's letters, it is a symmetry of C. For the outer product
 * @c "i,j,a,b <- i,a ; j,b" of t1 with itself that is the exchange of (i,a) with (j,b), and for
 * @c "ij <- ki ; kj" it is the swap of i and j. With exactly one operand conjugated, the swap
 * conjugates the term instead, and the generator is a conjugating one.
 *
 * This is the product alone. Whether the einsum's OUTPUT has the symmetry also depends on its
 * destination prefactor, which @ref output_symmetry and @ref symmetry_at account for.
 *
 * Declines on two different tensors in the two slots (even two views of one buffer, which hold
 * different elements), operands of different rank, a relabeling that is not consistent or does not
 * keep C's letters among C's, an output above @c kMaxSymmetryRank, a term wrapped in a
 * permutation operator, and a conjugating generator with a prefactor that is not real.
 *
 * @param[in] graph The graph @p node belongs to.
 * @param[in] node  The node.
 * @return A single-generator descriptor, or nothing.
 */
[[nodiscard]] EINSUMS_EXPORT std::optional<SymmetryDescriptor> self_product_symmetry(Graph const &graph, Node const &node);

/**
 * @brief The symmetry @p tensor has when @p reader reads it.
 *
 * A tensor written several times is still symmetric at a given read when the write that set its
 * contents produced the symmetry and every write after it, up to the read, added something that
 * has it too. The amplitude intermediate tau of a coupled-cluster iteration is that shape: an
 * outer product of t1 with itself, then t2 added in place. A single-writer guard cannot accept it;
 * this walks the writes in program order instead:
 *
 *  - a write with a zero destination prefactor sets the state to what it writes (an einsum's
 *    @ref self_product_symmetry, an axpby's source as it stands at that write);
 *  - a write that accumulates keeps only the generators both the state and its term have;
 *  - a scale keeps the state;
 *  - any other write, a write through another view of the buffer, a write by a descendant
 *    sub-graph, or an accumulation onto contents nothing before it set, gives no answer.
 *
 * With no write before @p reader, the answer is the tensor's own @c symmetry_hint: a symmetry the
 * user declared, which is a promise about every value the tensor holds, or one an analysis pass
 * proved for its final contents. That is how a loop-carried amplitude such as t2 enters.
 *
 * @pre The graph's node list is in an order that respects its dependencies, as after capture or
 *      @c Graph::topological_sort. "Before @p reader" means earlier in that list.
 *
 * @param[in] escape The analysis of the graph @p reader belongs to.
 * @param[in] tensor The tensor read.
 * @param[in] reader The node that reads it.
 * @param[out] relied_on When given, receives every tensor whose own @c symmetry_hint the answer
 *             rests on, with that hint: the promises a caller may want to check at run time.
 * @return The generators that hold at the read, or nothing.
 */
[[nodiscard]] EINSUMS_EXPORT std::optional<SymmetryDescriptor>
                             symmetry_at(EscapeAnalysis const &escape, TensorId tensor, NodeId reader,
                                         std::vector<std::pair<TensorId, SymmetryDescriptor>> *relied_on = nullptr);

/**
 * @brief Is what @p writer adds to its output unchanged, up to a sign, when its output's axes are
 *        permuted by @p sigma?
 *
 * The question a pass asks before computing half of a contraction and permuting the rest from it.
 * It is asked of the CONTRIBUTION, the term @p writer adds, not of the output tensor, which may
 * hold anything else as well.
 *
 * An operand of @p writer that exactly one einsum writes, by overwriting it, is replaced by that
 * einsum's own operands, and so on down, so a chain of contractions is judged as the one
 * expression it computes. The density-fitted particle ladder, @c H = B tau then @c R += H B, is
 * judged as @c R += B tau B. The remaining factors are then matched against their own images: the
 * contribution is unchanged when some relabeling of the summed letters, together with @p sigma on
 * the output letters, carries every factor onto a factor of the same tensor, up to an element of
 * that tensor's symmetry group as it stands where it is read (@ref symmetry_at). On the ladder the
 * relabeling exchanges e and f, the two B factors trade places, and tau's (i,a)-(j,b) exchange
 * absorbs the rest.
 *
 * Proof, not evidence: every step is an identity the operands satisfy, so a positive answer holds
 * for all values. A negative one may be conservative: the search takes generators as declared and
 * gives up on anything conjugated.
 *
 * @param[in] escape The analysis of the graph @p writer belongs to.
 * @param[in] writer An einsum node.
 * @param[in] sigma  The permutation of the output's positions, in the convention of
 *                   @c SymmetryOp: position @c p takes the letter at @c sigma.permutation[p]. Its
 *                   sign and conjugate fields are ignored.
 * @param[out] relied_on When given, receives the declared symmetries the proof rests on, as
 *             @ref symmetry_at reports them, each tensor once.
 * @return @p sigma with the sign the contribution picks up (+1 symmetric, -1 antisymmetric), or
 *         nothing when no proof was found.
 */
[[nodiscard]] EINSUMS_EXPORT std::optional<SymmetryOp>
                             contribution_symmetry(EscapeAnalysis const &escape, NodeId writer, SymmetryOp const &sigma,
                                                   std::vector<std::pair<TensorId, SymmetryDescriptor>> *relied_on = nullptr);

EINSUMS_NAMESPACE_END(compute_graph)
