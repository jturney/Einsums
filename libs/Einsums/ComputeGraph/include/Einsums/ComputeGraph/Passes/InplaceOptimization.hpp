//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief In-place storage merging for elementwise consumers.
 *
 * When an element-aligned elementwise node (DirectProduct, DirectDivision,
 * Axpby with beta == 0) pure-overwrites a graph-owned intermediate while
 * reading another graph-owned intermediate for the LAST time, the output
 * reuses the dying input's storage: the graph metadata merges the two ids
 * (CSE-style rewrite), the output's lifecycle nodes are dropped, and its
 * executor slot is durably redirected. One buffer allocation and its
 * write-traffic disappear per merge - the CC amplitude-update pattern
 * (R -> Tnew = R * invD) is the canonical win.
 *
 * @par Soundness guards
 * - Consumer whitelist: only ops whose out[i] depends solely on element i of
 *   the inputs may alias output with input (contractions and permutes never
 *   qualify). Pure overwrite is detected via the out-tensor-as-input
 *   recording convention.
 * - Both tensors: graph-owned intermediates, not viewed by anyone, identical
 *   dims and byte size; the source has exactly one producer and dies at the
 *   consumer; the destination has exactly one writer and no Initialize.
 * - Graphs containing control flow at this level are skipped (bodies
 *   reference parent tensors invisibly to plain use-counts); bodies are
 *   processed on their own recursion level. GPU-placed graphs are skipped
 *   (device shadows swap buffers behind the slots).
 *
 * num_candidates() still reports the single-producer/single-consumer census
 * this pass exposed when it was analysis-only.
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
