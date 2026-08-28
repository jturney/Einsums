//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file ProvenancePropagation.hpp
 * @brief Carry a tensor's provenance tag across the operations that preserve its identity.
 *
 * @par What propagates, and what deliberately does not
 * A tag says what a tensor IS. It travels across an operation only when the result is still the
 * same mathematical object:
 *
 * - **Permute and Transpose** carry it. The transpose of an identity is an identity; a
 *   reordering of an integral tensor's axes is still that integral tensor.
 * - **Nothing else does.** A contraction of two integrals is not an integral. A scaled delta is
 *   not a delta. A sum is not either of its terms.
 *
 * @par Why a VIEW does not carry it, which the obvious rule gets wrong
 * "Views, permutes and copies preserve identity" is the natural statement of this rule and it is
 * false for the tag that matters most here. A view of a Kronecker delta is an identity only when
 * the slice is a square block on the diagonal: ``delta[0:2, 0:4]`` is a perfectly ordinary
 * rectangular matrix of ones and zeros, and eliminating a contraction against it would produce a
 * wrong number rather than a slower one.
 *
 * The safe rule is therefore whole-tensor: an operation propagates a tag only when its output
 * covers exactly the same elements as its input. That is conservative for tags where slicing
 * WOULD be harmless - a sub-block of an integral tensor is still integral data, and this pass
 * will not say so - and the trade is deliberate. A tag that is missing costs a pass one
 * candidate; a tag that is wrong costs a wrong answer, and no later pass can tell the difference.
 * When a tag arrives whose slices genuinely inherit it, the rule gains a per-tag exception rather
 * than losing its default.
 *
 * @par An inferred tag never overwrites a declared one
 * A declaration is authoritative. If a caller has tagged an output, this pass leaves it alone
 * even when the producing node's input carries something else, and reports the disagreement
 * through its skip tally rather than picking a winner silently.
 *
 * @see TensorHandle::tag
 * @see Graph::annotate_tag
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Propagate provenance tags across identity-preserving operations.
 *
 * An analysis pass: it writes annotations and never touches the node set. See the file comment
 * for exactly which operations propagate and why a view is not one of them.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ProvenancePropagation : public OptimizerPass {
  public:
    /// @brief Default-construct. Explicit so the binding codegen has a constructor to annotate.
    APIARY_EXPOSE ProvenancePropagation() = default;

    /// @brief The pass name.
    /// @return ``"ProvenancePropagation"``.
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string name() const override { return "ProvenancePropagation"; }

    /// @brief Writes annotations only.
    /// @return @ref PassPhase::Analysis.
    [[nodiscard]] PassPhase phase() const override { return PassPhase::Analysis; }

    /// @brief Correct on a flat sub-graph, so loop bodies are visited too.
    /// @return True.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /**
     * @brief Propagate tags across @p graph.
     * @param[in,out] graph The graph to annotate.
     * @return False always: an analysis pass never changes the node set.
     */
    bool run(Graph &graph) override;

    /// @brief Zero the per-apply counters.
    void reset_stats() override { _num_propagated = 0; }

    /// @brief What this pass did, for @ref PassManager::explain.
    /// @return One line when it propagated anything, empty otherwise.
    [[nodiscard]] std::vector<std::string> explain() const override;

    /// @brief How many tensors received a tag on the last apply.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_propagated") [[nodiscard]] std::size_t num_propagated() const { return _num_propagated; }

  private:
    std::size_t _num_propagated{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
