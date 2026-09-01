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
 * @par What it declines
 * A delta whose two letters are both free in the output, or both contracted, is not a rename:
 * the first is a diagonal extraction and the second a trace, and both are real arithmetic this
 * pass does not do. A delta whose surviving letter already appears in the other operand would
 * rename two distinct letters into one, turning a contraction into a diagonal. Every one of
 * these is reported through the skip tally rather than attempted.
 *
 * @see RegionRewrite.hpp for the framework this is a client of
 * @see TensorHandle::tag
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Passes/RegionRewrite.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
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
    std::size_t _num_eliminated{0};
    std::size_t _num_dissolved{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
