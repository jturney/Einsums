//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief FreeInsertion pass: insert Free nodes to release intermediate tensors after their last consumer.
 *
 * For each graph-owned intermediate (is_intermediate == true, with a release_fn, not a view), the
 * last node that reads or writes it is found via the shared UsageAnalysis and a Free node is inserted
 * immediately after. The Free node calls release_fn() on the TensorHandle, freeing the backing storage
 * and returning the tensor to a deferred-like state; it lists the tensor as BOTH input and output so a
 * concurrent executor orders it after every prior reader and writer, not merely by node position.
 *
 * On re-execution the storage must reappear: deferred tensors already carry a Materialize node from the
 * Materialization pass, but eager create_tensor() intermediates do not, so this pass pairs a Materialize
 * before the tensor's first real writer for them (idempotent: a no-op on the first execute, reallocating
 * on every replay after the Free reclaimed the buffer).
 *
 * A body-resident intermediate (declared inside a loop body / conditional branch) is live across every
 * iteration, so its single Free is hoisted into the parent immediately after the enclosing control-flow
 * node - never inside the body, which would free-then-reuse each iteration.
 *
 * This pass is in the default pipeline, after InplaceOptimization (whose merges shorten the intervals it
 * works with) and before MemoryPlanning.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("free_insertion");
 * auto     &big = graph.scratch<double, 2>("big", n, n);    // graph-owned intermediate (>= 64 KiB)
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &big, 1.0, A, B);    // big produced
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, big, D);    // big last consumed here
 * }
 * graph.apply(cg::PassManager::create_default());           // FreeInsertion runs here
 * // A Free node now sits right after the second einsum, capping peak memory.
 * @endcode
 *
 * @par Example (Python)
 * Pipeline-internal: not exposed for standalone construction in Python. The default manager inserts the
 * Free (and any paired Materialize) for intermediates of at least 64 KiB.
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g  = cg.Graph("free_insertion")
 * ws = cg.Workspace("ws")
 * big = ws.declare_zero_tensor("big", [n, n], "float64")    # deferred intermediate
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", big, A, B)
 *     einsums.einsum("ij <- ik ; kj", C, big, D)
 * g.apply(cg.default_pass_manager())                        # frees `big` after its last consumer
 * @endcode
 *
 * @par Limitations
 * - Leaves held at most min_bytes (default 64 KiB) of tensors in all, the smallest first, and frees
 *   every other. One tensor under the floor saves a peak that is noise beside the two lifecycle nodes
 *   its Free costs every replay, but a graph may have any number of them, so the floor bounds their
 *   total, not each one. Allocation itself is not the reason: storage comes from mimalloc, which
 *   hands a freed block of the same size straight back.
 * - Frees graph-owned is_intermediate tensors, and graph-scoped tensors with a read backing: those
 *   are filled from their storage by an Initialize node on every replay, so holding them past their
 *   last use keeps a copy the next replay reads again anyway. After execute() such a tensor holds no
 *   storage; its value is the one in the backing. User-provided inputs and outputs of the whole
 *   computation, write and update backed tensors, workspace and pipeline tensors, and views are
 *   never freed.
 * - A body-resident intermediate's single Free is hoisted to AFTER the outermost enclosing loop, so it
 *   stays live across all iterations rather than being freed and reallocated per iteration.
 * - Idempotency across repeated runs is label-based (a hoisted Free does not carry the foreign child
 *   TensorId as an input), matching on "free(<name>)" / "materialize(<name>)".
 *
 * @par Future improvements
 * - The min_bytes floor is a fixed heuristic; a model of peak-memory saved against per-replay node
 *   cost could decide per tensor instead of a single global threshold.
 */
class EINSUMS_EXPORT FreeInsertion : public OptimizerPass {
  public:
    /// The default @p min_bytes: 64 KiB.
    static constexpr size_t default_min_bytes = static_cast<size_t>(64) * 1024;

    /// @param min_bytes The most bytes of tensors left held rather than freed, in all (default
    ///                  64 KiB), the smallest first.
    explicit FreeInsertion(size_t min_bytes = default_min_bytes) : _min_bytes(min_bytes) {}

    [[nodiscard]] std::string name() const override { return "FreeInsertion"; }
    /// Every feature: this pass moves, schedules, allocates or deletes nodes by dataflow and never reads their arithmetic.
    [[nodiscard]] std::optional<NodeFeatures> understood_features() const override { return NodeFeatures::all(); }

    /// @copydoc OptimizerPass::phase
    [[nodiscard]] PassPhase                phase() const override { return PassPhase::Tuning; }
    [[nodiscard]] std::vector<std::string> explain() const override;
    void                                   reset_stats() override;

    [[nodiscard]] size_t num_freed() const { return _num_freed; }

  protected:
    bool transform(Graph &graph) override;

  private:
    size_t _min_bytes;
    size_t _num_freed{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
