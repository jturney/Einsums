//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file DeltaElimination.hpp
 * @brief Contraction against a Kronecker delta is a rename, so do the rename and drop the delta.
 *
 * @par What it does
 * @code
 * tmp[i,j] = A[i,k] delta[k,j]        ->   (nothing; consumers read A directly)
 * C[i,l]  += tmp[i,j] D[j,l]          ->   C[i,l] += A[i,j] D[j,l]
 * @endcode
 *
 * Two nodes and one intermediate become one node. Machine-generated input - spin-summed
 * equations in particular - produces these in bulk, and each one is a full-size GEMM against an
 * identity matrix, so the win is a whole contraction rather than a constant factor on one.
 *
 * @par Why this is bitwise-exact rather than merely close
 * ``sum_k A[i,k] * I[k,j]`` has exactly one nonzero term. Every other product is
 * ``A[i,k] * 0.0``, which is exactly ``0.0``, and adding exact zeros into a running sum changes
 * nothing at all. The result is the same float, not a nearby one, so the rewrite needs no
 * tolerance and no opt-in.
 *
 * @par The one place that claim fails, deliberately
 * If any ``A[i,k]`` with ``k != j`` is infinite or NaN, the original computes ``0.0 * inf``,
 * which is NaN, and the sum is poisoned; the rewritten form returns ``A[i,j]`` unharmed. So the
 * two forms genuinely differ on non-finite input.
 *
 * This is accepted rather than guarded, on the grounds that the difference is one-directional:
 * the rewrite only ever REMOVES a NaN the arithmetic never had a reason to produce, and can
 * never introduce one. A pass that made results worse on degenerate input would be a different
 * question. Stated here rather than discovered later, because the differential fuzz shard
 * compares bitwise and does not skip degenerate programs, so this asymmetry is reachable by
 * design rather than by accident.
 *
 * @par Recognition is DECLARED, never read from the data
 * A tensor is a delta because someone said so, through @ref Graph::annotate_tag with
 * @ref provenance_identity. The alternative - inspecting the tensor's contents at pass time and
 * noticing it happens to be an identity - was considered and rejected. This is a
 * structural-algebraic pass, so its output is what a saved graph keeps, and a later bind may
 * supply a different tensor under the same manifest name. A rewrite justified by today's
 * contents would be written into a file and wrong on the next problem, which is precisely what
 * the phase rule exists to prevent.
 *
 * @par The second rewrite: a contraction that sums over nothing
 * @code
 * graph.space_registry().declare_disjoint(occ, virt);
 * graph.annotate_spaces(A, {any, occ});      // A's 'k' ranges over occ
 * graph.annotate_spaces(B, {virt, any});     // B's 'k' ranges over virt
 * C[i,j] = A[i,k] B[k,j]              ->     C[i,j] = 0
 * @endcode
 *
 * A letter summed over two spaces that share no element has no term to sum, so the contraction
 * contributes exactly nothing and what is left is the destination's own prefactor. The node
 * becomes a @c Scale of the destination by that prefactor, and disappears entirely when the
 * prefactor is zero and nothing reads the destination.
 *
 * Scaling by exactly zero ASSIGNS zero rather than multiplying, which is what makes the
 * overwriting case exact on a destination the program never wrote: a multiply would let an @c Inf
 * already in the buffer survive as a @c NaN.
 *
 * @par Where the exactness claim comes from, and the same hole
 * Both rewrites rest on a DECLARATION rather than on the data, and both are exact given that the
 * declaration is true. `sum_k A[i,k] I[k,j]` is exact because every other product is an exact
 * zero; `sum_{k in occ and virt} A[i,k] B[k,j]` is exact because the sum has no terms at all. And
 * the same one-directional hole applies: the captured form multiplies through values a rewritten
 * form never touches, so a non-finite operand poisons the sum where the rewrite returns a clean
 * number. The rewrite only ever removes a NaN and can never introduce one.
 *
 * @ref CrossSpaceValidation reports the same fact as an error, on the grounds that a contraction
 * over disjoint spaces is essentially never what an author meant. The two are one observation with
 * two responses, and running both is the intended combination: one says the result is zero and the
 * other says you probably did not mean to write it.
 *
 * @par What it declines
 * A delta whose two letters are both free in the output, or both contracted, is not a rename:
 * the first is a diagonal extraction and the second a trace, and both are real arithmetic this
 * pass does not do. A delta whose surviving letter already appears in the other operand would
 * rename two distinct letters into one, turning a contraction into a diagonal.
 *
 * For the zero half: a shared letter annotated on one operand and not the other, a shared letter
 * whose spaces nothing declared relates, a letter over disjoint spaces that is BATCHED rather than
 * summed (which makes the pairing meaningless, not the answer zero), a shared letter that repeats
 * within an operand and so has no single annotated slot, and disjointness resting on an annotation
 * @ref SpacePropagation INFERRED rather than one somebody declared. Every one of these is reported
 * through the skip tally rather than attempted.
 *
 * @see RegionRewrite.hpp for the framework this is a client of
 * @see CrossSpaceValidation.hpp for the diagnostic over the same fact
 * @see TensorHandle::tag
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Passes/RegionRewrite.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Eliminate contractions against a tagged Kronecker delta by index substitution.
 *
 * The first real client of the region rewrite framework: unlike @ref RegionIdentity it changes
 * the statement list, and unlike a peephole it decides what to do with the result by asking the
 * escape analysis whether the delta contraction's output can be dissolved.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT DeltaElimination : public RegionRewrite {
  public:
    /// @brief Default-construct. Explicit so the binding codegen has a constructor to annotate.
    APIARY_EXPOSE DeltaElimination() = default;

    /// @brief The pass name.
    /// @return ``"DeltaElimination"``.
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string name() const override { return "DeltaElimination"; }

    /// @copydoc OptimizerPass::tier
    /// Index substitution against a Kronecker delta removes multiplications by one; it does not reorder a sum.
    /// This pass is the reason PassTier::BitwiseExact carries its non-finite qualification: removing a multiply also
    /// removes its ability to produce a NaN, and that difference is one-directional.
    [[nodiscard]] PassTier tier() const override { return PassTier::BitwiseExact; }

    /// @brief How many delta contractions the last run removed.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_eliminated") [[nodiscard]] std::size_t num_eliminated() const { return _num_eliminated; }

    /// @brief How many of those also dissolved their output tensor entirely.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_dissolved") [[nodiscard]] std::size_t num_dissolved() const { return _num_dissolved; }

    /// @brief How many contractions summed over provably disjoint spaces and so were reduced to
    ///        their destination's own prefactor.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_zero_blocks") [[nodiscard]] std::size_t num_zero_blocks() const { return _num_zero_blocks; }

    /// @brief Zero the per-apply counters.
    void reset_stats() override;

  protected:
    /// @brief This pass's own line, appended to the framework's region report.
    /// @return One line when anything was eliminated, empty otherwise.
    [[nodiscard]] std::vector<std::string> describe() const override;

    /**
     * @brief Substitute away every delta contraction in one region's algebra.
     * @param[in]     graph  The graph, for tag lookups the algebra does not carry.
     * @param[in]     region The region being offered.
     * @param[in,out] expr   The algebra, rewritten in place.
     * @return True when anything was eliminated.
     */
    bool rewrite(Graph &graph, Region const &region, TensorExpr &expr) override;

    /**
     * @brief Does @p graph hold a tensor declared to be an identity?
     * @param[in] graph The graph.
     * @return True when at least one tensor carries @ref provenance_identity.
     *
     * This pass is in the default pipeline, so it runs on every graph anyone optimizes and
     * almost none of them declare a delta. One scan of the tensor map is what keeps that from
     * costing a region formation and a raise per region for a guaranteed no-op.
     */
    [[nodiscard]] bool applicable(Graph const &graph) const override;

  private:
    /**
     * @brief The summed letter that makes @p term identically zero, if it has one.
     *
     * @param[in] graph     The graph, read for its space registry and its CURRENT annotations.
     * @param[in] expr      The algebra, read for the operands' tensors.
     * @param[in] term      The contraction. Two operands, checked by the caller.
     * @param[in] statement The statement it is the value of.
     * @return The letter, or nothing. Everything it could not prove goes through the skip tally.
     */
    [[nodiscard]] std::optional<std::string> zero_block_letter(Graph const &graph, TensorExpr const &expr, ExprTerm const &term,
                                                               ExprStatement const &statement) const;

    std::size_t _num_eliminated{0};
    std::size_t _num_dissolved{0};
    std::size_t _num_zero_blocks{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
