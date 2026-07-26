//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief Lower a tiled einsum into per-tile DENSE nodes.
 *
 * A tiled contraction captures as one opaque ``OpKind::Custom`` node executed
 * tile-by-tile by ``detail::tiled_runtime_einsum``. Nothing can optimize it: the
 * einsum-rewriting passes all filter on ``OpKind::Einsum``, and a whole-tiled node
 * could not satisfy them anyway, since GEMMBatching wants one ``lda`` and
 * MemoryPlanning's arena wants one buffer, neither of which a tile container has.
 *
 * This pass replaces that node with one ordinary dense einsum per contributing
 * tile combination, built through @ref Graph::make_einsum_node. Because the result
 * is ordinary dense nodes, every existing pass then applies unchanged: the tile
 * GEMMs are same-shape so GEMMBatching can batch them, CSE deduplicates repeated
 * tile work, MemoryPlanning can pack per-tile buffers (each tile IS a dense
 * ``RuntimeTensor``, so it has ``materialize_into``), and Reorder schedules them.
 *
 * @par Semantics it must reproduce exactly
 * Ground truth is ``detail::tiled_runtime_einsum``:
 * - **Alignment.** Every shared index letter must carry an identical tile
 *   partition on each operand. The runtime throws; this pass DECLINES instead,
 *   leaving the opaque node, because the fallback is correct and merely
 *   unoptimized.
 * - **Sparsity.** A tile pair contributes only when both operand tiles are
 *   present, so absent tiles are structural zeros and emit nothing. Output tile
 *   ``C(c)`` exists iff some contributing combination reaches it.
 * - **The output prefactor applies exactly once.** The runtime pre-scales
 *   pre-existing output tiles by ``c_pf`` and then accumulates every contribution
 *   with ``beta = 1``. Expanded, the FIRST node writing a pre-existing tile
 *   carries ``c_pf`` and later ones carry 1; the first node writing a tile this
 *   pass CREATED carries 0, a pure overwrite, so correctness does not depend on
 *   the new tile being zero-initialized.
 * - **Pre-existing output tiles that receive NO contribution are still scaled.**
 *   The runtime scales them up front regardless. Emitting nodes only for
 *   contributing pairs would leave them untouched, which is a silent numerical
 *   difference, so this pass emits a scale (or a zero when ``c_pf == 0``) for
 *   them.
 *
 * @par Whole-tensor references must not be stranded
 * Expanding a node replaces its whole-tensor reads and writes with per-tile ones,
 * so the whole-tensor TensorId loses every reference that node owned. A node left
 * behind that still names that id would have its dependency edge silently
 * dropped, and with no writer a reader can be scheduled before the tiles are
 * filled. A candidate therefore expands only when EVERY node touching its tiled
 * tensors also expands; since rejecting one candidate can strand another, this
 * iterates to a fixpoint. Deciding this before any tile is created matters,
 * because minting a per-tile id creates the tile, and a spurious tile changes how
 * the runtime applies ``c_pf`` on the path that ends up not expanding.
 *
 * @par Limits
 * - Opt-in: not in @ref PassManager::populate_default.
 * - **Node budget.** Expansion produces up to (tiles of A) x (tiles of B) nodes,
 *   and per-node graph bookkeeping is on the order of microseconds, so a large
 *   grid can cost more in overhead than the contraction saves. The pass declines
 *   above @p max_nodes rather than guessing. A gate cannot be subtly wrong the way
 *   a cost heuristic can.
 * - **Operand tiles must already exist.** Sparsity is decided at pass time, so a
 *   tiled operand written by another node in the same graph is declined: its tile
 *   set is not known until execution. Tiled tensors built before capture, the
 *   normal usage, are fine.
 * - Creating the predicted output tiles is a side effect of ``apply()`` on user
 *   data, earlier than the execute-time infer-and-create it replaces.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("tiled");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", &C_tiled, A_tiled, B_tiled);  // one Custom node
 * }
 * cg::PassManager pm;
 * pm.add(std::make_shared<cg::passes::TiledExpansion>());
 * graph.apply(pm);   // now one dense Einsum per contributing tile pair
 * @endcode
 */
class EINSUMS_EXPORT TiledExpansion : public OptimizerPass {
  public:
    /// @param max_nodes Decline to expand when the projected node count exceeds this.
    explicit TiledExpansion(size_t max_nodes = 4096);

    [[nodiscard]] std::string name() const override { return "TiledExpansion"; }
    bool                      run(Graph &graph) override;
    void                      reset_stats() override;

    /// Safe per sub-graph: a tiled einsum is expanded within the single graph it
    /// is handed, and the nodes it emits stay in that graph.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Tiled einsum nodes replaced by per-tile nodes.
    [[nodiscard]] size_t num_expanded() const { return _num_expanded; }
    /// Dense per-tile nodes emitted, contractions and scales together.
    [[nodiscard]] size_t num_tile_nodes() const { return _num_tile_nodes; }
    /// Candidates left alone: misaligned partitions, produced operands, over
    /// budget, or sharing a tiled tensor with a node that cannot expand.
    [[nodiscard]] size_t num_declined() const { return _num_declined; }

  private:
    size_t _max_nodes;
    size_t _num_expanded{0};
    size_t _num_tile_nodes{0};
    size_t _num_declined{0};
};

} // namespace einsums::compute_graph::passes
