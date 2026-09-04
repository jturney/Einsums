//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file LayoutAssignment.hpp
 * @brief Storage order as a graph-wide decision variable, chosen so contractions read their
 *        operands flat.
 *
 * @par The quantity being minimized
 * A vendor GEMM reads A as a flat @c (M,K) matrix and B as a flat @c (K,N) one. An operand
 * whose contracted letters sit at one END of its index list has such a reading already, and the
 * transpose flag settles which end (@ref link_placement is the library's own statement of that
 * rule, and @ref LinkPlacement::split names the remaining case). An operand whose
 * contracted letters are INTERLEAVED with its free ones has no flat reading at all, so the
 * kernel copies it into one before it can multiply. That copy is a whole tensor of traffic per
 * replay, and it is the only cost in a contraction that storage order decides.
 *
 * A graph-owned intermediate has no storage order anyone promised. Its axes exist in the order
 * whoever captured the program happened to write them, and every node that touches it can be
 * told to index it differently. So the order is free to choose, and choosing it well is what
 * this pass does.
 *
 * @par Global, which is the whole point
 * @ref PermuteFusion is the peephole form of this: one permute, one consumer, absorbed. It
 * declines the moment a permuted tensor has two readers, because a peephole has no way to
 * decide between them. This pass has: it prices every reading of every candidate under the
 * @ref CostModel and takes the assignment with the lowest total, so a tensor with three
 * consumers that disagree gets the layout the expensive one wanted and the cheap ones pay.
 *
 * @par An explicit permute is a layout question too
 * A @c Permute node writes a copy of its source in a different axis order. Storing that copy the
 * way its source is stored makes the node an identity, and an identity copy is a node to delete
 * rather than one to run, so the permute joins the same cost total as the contractions: it is
 * priced at one tensor-sized copy while it survives and at nothing once it is gone. That makes
 * the deletion the OUTCOME of a layout choice rather than a separate peephole, and it reaches the
 * case a peephole cannot, a permuted copy several contractions read.
 *
 * The copy's source keeps the axes it was captured with. A permute's index lists are a structural
 * field an executor bakes at build time, so a permute this pass does not delete is one it must
 * leave reading exactly what it read, which leaves the copy two orders to choose between: the
 * captured one, and the one that deletes the node.
 *
 * @code
 * // W is captured as (m,b,e,j). The two consumers want opposite groupings, and
 * // whichever one the capture happened to suit, the other one copies.
 * cg::einsum("mbej <- mnef ; njfb", 0.0, &W, 1.0, I, T);   // producer
 * cg::einsum("ijab <- imae ; mbej", 1.0, &R, 1.0, T2, W);  // reader, wants (m,e) adjacent
 * // Assigned (b,j,m,e): both contractions then read W flat, and no copy is made.
 * @endcode
 *
 * @par What is a decision variable and what is not
 * Only tensors of rank three or more, owned by the graph, still deferred, and unobservable from
 * outside it (@ref EscapeAnalysis reports @ref Escape::Dissolvable). Each exclusion earns its
 * place:
 * - **Rank two is already free.** A matrix has two readings and BLAS takes either one through
 *   @c transa, so there is nothing to win and a @ref GemmHint to invalidate. Skipping rank two
 *   is also what keeps this pass away from every hint in the graph, since a hint exists only for
 *   an all-rank-2 contraction.
 * - **A user's tensor keeps the user's axes.** Its order is part of what the caller asked for,
 *   and a manifest binds it by name at a shape the caller chose.
 * - **Deferred, because a materialized intermediate holds bytes.** Re-laying out an unallocated
 *   shell is a change of declaration; re-laying out a live buffer is a data movement this pass
 *   does not perform.
 * - **Unobservable, because relaying out a tensor rewrites every index list that names it.** A
 *   reader this graph cannot see is a reader that would keep indexing the old order.
 *
 * A tensor is also pinned by any use this pass cannot rewrite: a node that is not a contraction
 * (beyond the lifecycle kinds, which do not index anything), an operand that repeats a letter,
 * or a contraction carrying a letter that is neither free, contracted, nor batched. Each is
 * counted in the skip tally rather than guessed at.
 *
 * @par The cost model is a hint, and the phase rule is why that matters
 * This is a structural-algebraic pass, so a save keeps its output. The permute-time estimates
 * it consults are a property of the machine that ran it, which makes this the one pass in that
 * phase with a machine-dependent input. What keeps it inside the rule is that the OUTPUT is
 * machine-independent: an index list is an index list, and a layout chosen from another
 * machine's permute costs computes the same numbers here, merely not necessarily the fastest
 * ones. Batching and thread widths are re-derived after it on every load, which is the phase
 * rule doing its job rather than an accident.
 *
 * @par Why the tier is re-associating and not bitwise
 * Nothing about the algebra changes: the same products are summed over the same letters. What
 * changes is which kernel the dispatcher reaches, and a flat GEMM accumulates in a different
 * order than a packed one. That is the same reasoning that puts @ref PermuteFusion in this tier,
 * and it is the reason a tier whose members each carry an exception has stopped drawing a line.
 *
 * @see EinsumSpec.hpp for @ref link_placement, the flat-reading test this shares with the executor
 * @see PermuteFusion.hpp for the peephole this generalizes
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Choose the storage order of every graph-owned intermediate so contractions read flat.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("layout");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("mbej <- mnef ; njfb", 0.0, &W, 1.0, I, T);
 *     cg::einsum("ijab <- imae ; mbej", 1.0, &R, 1.0, T2, W);
 * }
 * cg::PassManager pm;
 * pm.add<cg::passes::LayoutAssignment>();
 * graph.apply(pm);   // W's axes are reordered and both index lists follow
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums.graph as cg
 * la = cg.LayoutAssignment()
 * pm = cg.PassManager(); pm.add(la); pm.run(g)
 * # la.num_relaid_out -> 1  (getters are properties, not methods)
 * @endcode
 * It also runs as part of ``cg.default_pass_manager()``.
 *
 * @par Limitations
 * - Candidates are rank-3-and-up graph-owned deferred intermediates only; see the file note for
 *   why each half of that is there.
 * - The search is coordinate descent over the layouts some participant asked for, not an
 *   exhaustive one over @c r! orders per tensor. A layout nobody wants cannot beat one that at
 *   least one node does, which is what makes the restriction defensible rather than merely
 *   cheap; it is not a proof of optimality, and the pass does not claim one.
 * - A @c Permute is folded away only when its copy is one of the decision variables above, which
 *   is where @ref PermuteFusion still has cases of its own: a rank-two copy, a copy the caller
 *   owns, a copy already allocated, and a copy read by something other than a contraction are
 *   each pinned here and each still absorbed there. A @c Transpose carries no descriptor and is
 *   rank two by definition, so it is never a candidate.
 * - The copy's source keeps its captured axes, so a chain of two permutes folds neither.
 * - A contraction with a repeated letter in one operand (a diagonal) is not modelled, and pins
 *   its tensors.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT LayoutAssignment : public OptimizerPass {
  public:
    /// @brief Construct against the detected hardware profile.
    APIARY_EXPOSE LayoutAssignment();

    /// @brief Price the copies against a supplied profile instead of the detected one.
    /// @param[in] cost_model The profile. ``populate_default()`` passes the one it shares with
    ///                       the other cost-model passes, so every decision is made against one
    ///                       machine.
    explicit LayoutAssignment(CostModel cost_model);

    /// @brief The pass name.
    /// @return ``"LayoutAssignment"``.
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string name() const override { return "LayoutAssignment"; }

    /// @copydoc OptimizerPass::phase
    /// Storage order is mathematics-preserving and machine-independent to state, so a save keeps
    /// it. The permute costs it is chosen from are a hint; see the file note.
    [[nodiscard]] PassPhase phase() const override { return PassPhase::StructuralAlgebraic; }

    /// @copydoc OptimizerPass::tier
    /// The same products summed over the same letters, through a different kernel. See the file
    /// note on why that is this tier and not the one above it.
    [[nodiscard]] PassTier tier() const override { return PassTier::ReAssociating; }

    /// Safe on loop bodies and conditional branches: every tensor it moves is one the escape
    /// analysis says nothing outside the graph it was handed can observe, and a sub-graph is
    /// such a graph.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    bool                                   run(Graph &graph) override;
    [[nodiscard]] std::vector<std::string> explain() const override;
    void                                   reset_stats() override;

    /// @brief How many intermediates were given an order other than the captured one.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_relaid_out") [[nodiscard]] std::size_t num_relaid_out() const { return _num_relaid_out; }

    /// @brief How many @c Permute nodes the chosen layout turned into identity copies and deleted.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_permutes_folded") [[nodiscard]] std::size_t num_permutes_folded() const {
        return _num_permutes_folded;
    }

    /// @brief How many operand copies the chosen assignment removes, per replay.
    /// @return The count, over every contraction the pass modelled.
    APIARY_EXPOSE APIARY_GETTER("num_copies_removed") [[nodiscard]] std::size_t num_copies_removed() const { return _num_copies_removed; }

    /// @brief What those copies were modelled to cost, in microseconds per replay.
    /// @return The estimate. A modelled number and not a measured one; see the file note.
    APIARY_EXPOSE APIARY_GETTER("estimated_saving_us") [[nodiscard]] double estimated_saving_us() const { return _estimated_saving_us; }

  private:
    CostModel   _cost_model;
    std::size_t _num_relaid_out{0};
    std::size_t _num_permutes_folded{0};
    std::size_t _num_copies_removed{0};
    double      _estimated_saving_us{0.0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
