//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>
#include <string>

namespace einsums::compute_graph::passes {

/**
 * @brief Fold the CCSD permutational-symmetrization idiom
 *        @f$r2 \mathrel{+}= s\,(t + P(t))@f$.
 *
 * Chemistry residuals accumulate a tensor and its index-swapped transpose into
 * the same output. In the graph this appears as four nodes per site (the
 * @c symacc helper in @c ccsd_rhf_graph_numpy_style.py):
 * @code
 * einsum(spec, A, B)            -> tmp    // tmp = pf*(A(x)B)   (kept)
 * axpby(s, tmp,  1.0, r2)                 // r2 += s*tmp
 * permute("jiba <- ijab", tmp) -> tmpP    // tmpP = P(tmp)   (an involution)
 * axpby(s, tmpP, 1.0, r2)                 // r2 += s*tmp^{jiba}
 * @endcode
 *
 * The pass detects each such site (structural match + involutive permutation +
 * accumulate gate + interference guard) and rewrites it (Level 1, permute
 * reuse): the permute node is made to accumulate DIRECTLY into the output -
 * @f$r2 = r2 + s_2\,P(t)@f$ via the existing string_permute kernel with beta = 1
 * - and the second axpby and the @c tmpP buffer are deleted. The first axpby
 * (@f$r2 \mathrel{+}= s_1 t@f$) is untouched. This drops the O(o^2 v^2) @c tmpP
 * storage and one full sweep per site with no new kernel or OpKind. See
 * @c docs/symmetrized_accumulation_design.md.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("symacc");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ijab <- ia ; jb", 0.0, &tmp, 1.0, A, B);  // tmp = A (x) B
 *     cg::axpby(0.5, tmp, 1.0, &r2);                         // r2 += 0.5*tmp
 *     cg::permute("jiba <- ijab", 0.0, &tmpP, 1.0, tmp);     // tmpP = P(tmp)
 *     cg::axpby(0.5, tmpP, 1.0, &r2);                        // r2 += 0.5*P(tmp)
 * }
 * graph.apply(cg::PassManager::create_default());  // fires on RuntimeTensor operands
 * // tmpP and the second axpby are gone; the permute now does r2 += 0.5*P(tmp).
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("symacc")
 * with cg.capture(g):                                     # runtime tensors (the workload path)
 *     einsums.einsum("i,j,a,b <- i,a ; j,b", tmp, A, B)   # tmp = A (x) B
 *     einsums.linalg.axpby(0.5, tmp, 1.0, r2)             # r2 += 0.5*tmp
 *     einsums.permute("j,i,b,a <- i,j,a,b", tmpP, tmp)    # tmpP = P(tmp)
 *     einsums.linalg.axpby(0.5, tmpP, 1.0, r2)            # r2 += 0.5*P(tmp)
 * sa = cg.SymmetrizedAccumulation()
 * pm = cg.PassManager(); pm.add(sa)
 * g.apply(pm)                                             # or cg.default_pass_manager()
 * # sa.num_rewritten  -> 1   (getters are properties, not methods)
 * @endcode
 *
 * In the default pipeline (after DeadNodeElimination, before ElementWiseFusion —
 * which would otherwise compose the two axpby and hide the pattern). Recurses
 * into loop bodies: the CCSD residual is captured as a loop body.
 *
 * @par Limitations
 * - Fires only on RUNTIME tensors of one dtype; typed @c Tensor<T,Rank> captures
 *   are left folded-out, and only when the matched permute's alpha == 1.
 * - Level 1 keeps the transpose (the accumulating permute), so REPLAY TIME is
 *   ~compute-neutral (a few %, transpose-bound). The win is eliminating the
 *   O(o^2 v^2) @c tmpP buffer (peak memory), not compute.
 * - Detection requires the exact structure: an involutive overwrite permute
 *   whose output is sole-produced/sole-consumed by one accumulating axpby, plus
 *   a sibling accumulating axpby reading the un-permuted source into the same
 *   output; an interference guard rejects sites where another node observes the
 *   half-symmetrized output.
 *
 * @par Future improvements
 * - Level 2: a fused single-sweep @c SymmetrizedAxpby kernel
 *   (@f$r2 \mathrel{+}= s_1 t + s_2 P(t)@f$ in one pass), profiling-gated — drops
 *   the second sweep as well, not just the buffer.
 * - Generalize beyond involutive permutations and allow @f$s_1 \neq s_2@f$.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT SymmetrizedAccumulation : public OptimizerPass {
  public:
    APIARY_EXPOSE SymmetrizedAccumulation() = default;

    [[nodiscard]] std::string name() const override { return "SymmetrizedAccumulation"; }

    bool run(Graph &graph) override;

    /// The CCSD residual is a loop body; the sites live in the subgraph.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Structural matches: permute (involutive, overwrite) whose output is the
    /// sole-produced/sole-consumed source of an accumulating axpby into the same
    /// output that a sibling accumulating axpby also feeds from the un-permuted
    /// source.
    APIARY_EXPOSE APIARY_GETTER("num_candidates") [[nodiscard]] size_t num_candidates() const { return _num_candidates; }

    /// Candidates that also pass the interference guard (no other node observes
    /// the half-symmetrized output or rewrites the source mid-fold) - the sites
    /// that are safe to fold.
    APIARY_EXPOSE APIARY_GETTER("num_matched") [[nodiscard]] size_t num_matched() const { return _num_matched; }

    /// Sites actually rewritten (num_matched that also passed the runtime-tensor
    /// / uniform-dtype gate). Each eliminates one axpby and one tmpP buffer.
    APIARY_EXPOSE APIARY_GETTER("num_rewritten") [[nodiscard]] size_t num_rewritten() const { return _num_rewritten; }

  private:
    size_t _num_candidates{0};
    size_t _num_matched{0};
    size_t _num_rewritten{0};
};

} // namespace einsums::compute_graph::passes
