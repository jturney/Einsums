//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Materialization pass: give every deferred (shell) tensor a Materialize (and optional
 *        Initialize) lifecycle node so it is allocated at the right point during execution.
 *
 * For each tensor with AllocState::Deferred, this pass:
 * 1. Finds the earliest node that writes to or reads from the tensor, via the shared
 *    UsageAnalysis (a view of a deferred tensor counts as a use of its owner, and a use buried
 *    in a loop body / conditional branch surfaces at the enclosing control-flow node's position).
 * 2. Inserts a Materialize node just before that point; its executor calls materialize_fn() to
 *    allocate the backing storage (for a distributed, non-replicated tensor it first resizes to
 *    this rank's local partition using the DistributionDescriptor that DistributionPlanning attached).
 * 3. If init_kind is set (Zero, Random), inserts an Initialize node immediately after.
 *
 * Allocation happens lazily during graph.execute() when the Materialize node runs, NOT during the
 * pass itself. So intermediates only consume memory when needed, MemoryPlanning can see the
 * Materialize nodes for accurate peak estimation, and distributed allocation happens per-rank at
 * execution time. A deferred tensor used inside a loop body or conditional branch has its lifecycle
 * HOISTED into the outermost parent (keyed and deduped by the underlying buffer pointer, so one
 * buffer is materialized + initialized exactly once) and thus runs once per outer execution instead
 * of every iteration / branch entry.
 *
 * This pass is CORRECTNESS-ENABLING, not an optimization: a graph that uses declare_tensor() /
 * scratch() cannot execute without it, so it is included at every optimization level above O0. It
 * is in the default pipeline, after DistributionPlanning and before SymmetryPropagation / the GPU
 * passes.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("materialization");
 * auto     &tmp = graph.scratch<double, 2>("tmp", n, n);    // deferred + intermediate
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &tmp, 1.0, A, B);    // tmp = A·B   (first use)
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, tmp, D);    // C = tmp·D
 * }
 * graph.apply(cg::PassManager::create_default());           // Materialization runs here
 * graph.execute();   // a Materialize node allocates `tmp` just before the first einsum
 * @endcode
 *
 * @par Example (Python)
 * Pipeline-internal: the pass is not exposed for standalone construction in Python. Deferred
 * (workspace) tensors are materialized through the default manager.
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g  = cg.Graph("materialization")
 * ws = cg.Workspace("ws")
 * tmp = ws.declare_zero_tensor("tmp", [n, n], "float64")    # deferred, zero-initialized
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", tmp, A, B)
 *     einsums.einsum("ij <- ik ; kj", C, tmp, D)
 * g.apply(cg.default_pass_manager())                        # materializes + zeroes tmp at first use
 * g.execute()
 * @endcode
 *
 * @par Limitations
 * - Acts only on tensors in AllocState::Deferred; eager create_tensor() tensors are left untouched.
 * - Allocation is deferred to graph.execute() (the Materialize node's executor), never performed by
 *   the pass itself.
 * - Body-scoped deferred tensors are hoisted to the outermost parent and materialized / initialized
 *   once per outer execution; the dedup key is the underlying tensor_ptr, so a null buffer pointer
 *   cannot be deduplicated and gets its own lifecycle pair.
 * - Distributed (non-replicated) tensors require their DistributionDescriptor to have been attached
 *   already by DistributionPlanning; the Materialize executor resizes to the per-rank local partition.
 */
class EINSUMS_EXPORT Materialization : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "Materialization"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_materialized() const { return _num_materialized; }
    [[nodiscard]] size_t num_initialized() const { return _num_initialized; }

  private:
    size_t _num_materialized{0};
    size_t _num_initialized{0};
};

} // namespace einsums::compute_graph::passes
