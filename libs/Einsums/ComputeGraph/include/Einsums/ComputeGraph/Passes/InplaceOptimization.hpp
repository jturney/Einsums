//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief InplaceOptimization pass: in-place storage merging for elementwise consumers.
 *
 * When an element-aligned elementwise node (DirectProduct, DirectDivision, Axpby with beta == 0)
 * pure-overwrites a graph-owned intermediate while reading another graph-owned intermediate for the
 * LAST time, the output reuses the dying input's storage: the graph metadata merges the two ids
 * (CSE-style rewrite), the output's lifecycle nodes are dropped, and its executor slot is durably
 * redirected. One buffer allocation and its write-traffic disappear per merge - the CC amplitude-update
 * pattern (R -> Tnew = R * invD) is the canonical win.
 *
 * This pass is in the default pipeline, run before FreeInsertion / MemoryPlanning so each merge removes
 * a buffer and shortens the intervals those liveness passes then work with.
 *
 * @par Soundness guards
 * - Consumer whitelist: only ops whose out[i] depends solely on element i of the inputs may alias
 *   output with input (contractions and permutes never qualify). Pure overwrite is detected via the
 *   out-tensor-as-input recording convention.
 * - Both tensors: graph-owned intermediates, not viewed by anyone, identical dims and byte size; the
 *   source has exactly one producer and dies at the consumer; the destination has exactly one writer
 *   and no Initialize.
 * - Graphs containing control flow at this level are skipped (bodies reference parent tensors invisibly
 *   to plain use-counts); bodies are processed on their own recursion level. GPU-placed graphs are
 *   skipped (device shadows swap buffers behind the slots).
 *
 * num_candidates() still reports the single-producer/single-consumer census this pass exposed when it
 * was analysis-only.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("inplace");
 * auto     &R    = graph.scratch<double, 2>("R", n, n);       // residual (dies at the update)
 * auto     &invD = graph.scratch<double, 2>("invD", n, n);    // 1/denominator
 * auto     &Tnew = graph.scratch<double, 2>("Tnew", n, n);    // amplitude update
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &R, 1.0, A, B);        // produce R
 *     cg::direct_product(1.0, R, invD, 0.0, &Tnew);           // Tnew = R (*) invD, R read last
 * }
 * graph.apply(cg::PassManager::create_default());             // InplaceOptimization runs here
 * // Tnew now reuses R's storage; R's buffer and its write-traffic are gone.
 * @endcode
 *
 * @par Example (Python)
 * Constructible standalone (getters are properties). A whitelisted elementwise consumer merges; an
 * einsum consumer never does (nothing merges, num_merged stays 0).
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("inplace")
 * R    = g.create_zero_tensor("R",    [n, n], dtype="float64")
 * invD = g.create_random_tensor("invD", [n, n], dtype="float64")
 * Tnew = g.create_zero_tensor("Tnew", [n, n], dtype="float64")
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", R, A, B)
 *     einsums.linalg.direct_product(1.0, R, invD, 0.0, Tnew)
 * p  = cg.InplaceOptimization()
 * pm = cg.PassManager(); pm.add(p); pm.run(g)                 # or g.apply(cg.default_pass_manager())
 * # p.num_merged, p.num_candidates  (properties, not methods)
 * @endcode
 *
 * @par Limitations
 * - Only the element-aligned elementwise consumers DirectProduct, DirectDivision and Axpby(beta == 0)
 *   may alias output with a dying input; contractions (Einsum/Gemm/BatchedGemm) and permutes are
 *   excluded and never merge.
 * - Both tensors must be graph-owned intermediates, non-viewed, with identical dims and byte size; the
 *   source needs exactly one producer and this node as its only reader, the destination exactly one
 *   writer and a pure overwrite (destination not read).
 * - HOST-only and single-level: any graph with control flow at this level, or with GPU placement /
 *   host<->device transfers, is skipped (bodies are handled on their own recursion level).
 * - A destination carrying an Initialize node is skipped rather than merged; the v1 pass does not
 *   delete the (now-dead) Initialize.
 *
 * @par Future improvements
 * - Handle a destination with an Initialize by deleting the dead init instead of bailing on the merge.
 * - Extend beyond the elementwise whitelist to any consumer whose access is provably element-aligned.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT InplaceOptimization : public OptimizerPass {
  public:
    APIARY_EXPOSE InplaceOptimization() = default;

    [[nodiscard]] std::string name() const override { return "InplaceOptimization"; }
    bool                      run(Graph &graph) override;

    APIARY_EXPOSE APIARY_GETTER("num_candidates") [[nodiscard]] size_t num_candidates() const { return _num_candidates; }

    /// Number of output buffers merged into dying inputs in the last run.
    APIARY_EXPOSE APIARY_GETTER("num_merged") [[nodiscard]] size_t num_merged() const { return _num_merged; }

  private:
    size_t _num_candidates{0};
    size_t _num_merged{0};
};

} // namespace einsums::compute_graph::passes
